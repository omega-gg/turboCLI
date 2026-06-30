#==================================================================================================
#
#   Copyright (C) 2026-2026 turboCLI authors. <https://omega.gg/turboCLI>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turboCLI.
#
#   - GNU Lesser General Public License Usage:
#   This file may be used under the terms of the GNU Lesser General Public License version 3 as
#   published by the Free Software Foundation and appearing in the LICENSE.md file included in the
#   packaging of this file. Please review the following information to ensure the GNU Lesser
#   General Public License requirements will be met: https://www.gnu.org/licenses/lgpl.html.
#
#   - Private License Usage:
#   turboCLI licensees holding valid private licenses may use this file in accordance with the
#   private license agreement provided with the Software or, alternatively, in accordance with the
#   terms contained in written agreement between you and turboCLI authors. For further information
#   contact us at contact@omega.gg.
#
#==================================================================================================

# HTTP front-end: a long-lived host that caches the model and serves many generations, with
# preemption and idle release. Same endpoints / urlencoded params / streamed "Saved:" protocol as
# the old server.sh, so every existing curl client keeps working unchanged. The per-generation work
# (load-or-reuse + run + save) lives in core.generate, shared with cli.py; this file owns only the
# server-specific concerns: HTTP, locks, the latest-wins preemption id, and the idle watcher.
#
# Run as: python -m runner.server [--host H] [--port P] [--timeout S] [--range N] [--scan]
# (server.sh passes these from its own settings).

import argparse
import time
import socket
import threading
import traceback

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

# NOTE: Importing core runs offload-backend discovery before torch, then imports torch. Nothing
#       above touches torch, so the pre_torch_init()-before-torch invariant holds.
from runner import core

log = core.log

_parser = argparse.ArgumentParser(prog="server", description="turboCLI generation server.")
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=8080)
# Clear the resident model after this many idle seconds (0 disables it).
_parser.add_argument("--timeout", type=float, default=600.0)
# Number of ports to try (starting at --port) when scanning for a free one.
_parser.add_argument("--range", type=int, default=20)
# Bind the first free port in [--port, --port + --range - 1] if the requested one is taken.
_parser.add_argument("--scan", action="store_true")
_args = _parser.parse_args()

HOST = _args.host
PORT = _args.port
TIMEOUT = _args.timeout
RANGE = _args.range

# Concurrency: requests run in their own threads (ThreadingHTTPServer) but only one generation
# touches the GPU at a time (gpu_lock). Each request takes the next latest_id; an in-flight job
# compares its id against latest_id at every step and aborts itself when a newer request arrives.
gpu_lock = threading.Lock()
state_lock = threading.Lock()
latest_id = 0

# Whether the most recent bump of latest_id was an explicit /cancel (vs a newer /generate).
last_was_cancel = False

# Time at which the server last became idle (no generation running). The idle watcher releases the
# resident model once this is older than TIMEOUT.
last_active = time.time()


def idle_watcher():
    if TIMEOUT <= 0:
        return

    while True:
        time.sleep(30)

        with state_lock:
            idle_for = time.time() - last_active

        if idle_for < TIMEOUT:
            continue

        # Skip (rather than block) when a generation is running; it'll be picked up on a later
        # tick.
        if not gpu_lock.acquire(blocking=False):
            continue

        try:
            core.release_pipe("idle for %.0fs" % idle_for)
        finally:
            gpu_lock.release()


class Handler(BaseHTTPRequestHandler):

    def setup(self):
        BaseHTTPRequestHandler.setup(self)

        # NOTE: Disable Nagle so each flushed progress line (and "Saved:") goes out at once.
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def _send(self, code, body):
        data = body.encode("utf-8")

        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, "ok")
        else:
            self._send(404, "not found")

    def do_POST(self):
        global latest_id, last_was_cancel, last_active

        if self.path.startswith("/shutdown"):
            self._send(200, "shutting down")

            threading.Thread(target=httpd.shutdown, daemon=True).start()

            return

        if self.path.startswith("/cancel"):
            # Bump the id (so the in-flight job aborts at its next step) without queuing work.
            with state_lock:
                latest_id += 1
                last_was_cancel = True

            busy = gpu_lock.locked()

            self._send(200, "cancelling current task\n" if busy else "idle, nothing to cancel\n")

            return

        if self.path.startswith("/clear"):
            # Cancel any task in progress (same as /cancel) so the GPU lock frees up.
            with state_lock:
                latest_id += 1
                last_was_cancel = True

            if not gpu_lock.acquire(timeout=30):
                self._send(200, "busy, could not clear (timed out waiting for current task)\n")

                return

            try:
                if core.release_pipe("on request"):
                    self._send(200, "model cleared\n")
                else:
                    self._send(200, "no model loaded\n")
            finally:
                gpu_lock.release()

            return

        if not self.path.startswith("/generate"):
            self._send(404, "not found")

            return

        length = int(self.headers.get("Content-Length", "0"))

        raw = self.rfile.read(length).decode("utf-8")

        fields = parse_qs(raw, keep_blank_values=True)

        params = {k: v[0] for k, v in fields.items()}

        # NOTE: We stream progress as a plain-text body. The status line is sent up front, so we
        #       cannot signal failure with an HTTP error code: the client treats a "Saved:" line as
        #       success and anything else as failure.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

        def emit(line):
            try:
                self.wfile.write((line + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                # Client went away (e.g. it timed out); keep going so the image still saves.
                pass

        # Claim the latest id; this makes any in-flight job abort itself at its next step.
        with state_lock:
            latest_id += 1
            last_was_cancel = False
            my_id = latest_id

        if gpu_lock.locked():
            emit("another job is in progress, asking it to stop...")

        gpu_lock.acquire()

        try:
            # A still-newer request may have arrived while we waited for the lock.
            with state_lock:
                stale = my_id != latest_id

            if stale:
                emit("SUPERSEDED: a newer request arrived before this one started")
            else:
                # latest-wins preemption: an in-flight job aborts when a newer request bumps the
                # id.
                def should_stop():
                    with state_lock:
                        if my_id == latest_id:
                            return None

                        return "cancel" if last_was_cancel else "supersede"

                prompt = params.get("prompt", "")

                log('generation started from %s: "%s" (%sx%s, %s, %s steps, %s)'
                    % (self.client_address[0], prompt[:60],
                       params.get("width", "?"), params.get("height", "?"),
                       params.get("engine", "?"), params.get("inference", "?"),
                       params.get("renderer", "?")))

                core.generate(params, emit, should_stop)
        except Exception:
            tb = traceback.format_exc()

            log(tb)

            try:
                self.wfile.write(("ERROR: " + tb + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
        finally:
            gpu_lock.release()

            with state_lock:
                last_active = time.time()

    def log_message(self, fmt, *args):
        # NOTE: Silence the default per-request access log, we keep our own.
        return


class Server(ThreadingHTTPServer):
    # NOTE: Threaded so a new request can be received while a generation is running; the actual GPU
    #       work is still serialized by gpu_lock, and a newer request preempts the old one.
    daemon_threads = True
    request_queue_size = 64
    allow_reuse_address = True


# Optional scan: if the requested port is taken, bind the first free one in
# [PORT, PORT + RANGE - 1]. The probe is a plain socket (no SO_REUSEADDR) so an in-use port
# reliably fails, including on Windows where the server's allow_reuse_address would otherwise let
# it bind a live port.
if _args.scan:
    for candidate in range(PORT, PORT + RANGE):
        probe = socket.socket()

        try:
            probe.bind((HOST, candidate))
        except OSError:
            continue
        finally:
            probe.close()

        PORT = candidate
        break
    else:
        raise SystemExit("No free port available in %d-%d" % (PORT, PORT + RANGE - 1))

httpd = Server((HOST, PORT), Handler)

log("server is running on http://%s:%d" % (HOST, PORT))
log("Model loads on the first request, and reloads when engine / model / renderer / offload /"
    " slicing change.")
log("A new request preempts the one in progress (latest wins); generations never run in parallel.")

if TIMEOUT > 0:
    log("Idle model gets released after %.0fs of inactivity." % TIMEOUT)

    threading.Thread(target=idle_watcher, daemon=True).start()

log("Press Ctrl-C to stop, or run: server stop")

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    httpd.server_close()

    log("Server stopped.")
