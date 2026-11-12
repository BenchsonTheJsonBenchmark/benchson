"""Three-way schema validation for the data-generation checks: VALID, INVALID or TECHNICAL.

`metrics.schema_compliance` (the evaluation API, unchanged) returns 0 both when an example violates its schema and when
validation could not be carried out at all (an external `$ref` that could not be fetched, a malformed schema, a crash).
For a benchmark check those are different facts: only the first is evidence about the example. Here:

    VALID      validation ran to the end and found no violation.
    INVALID    validation ran to the end against a fully resolvable, well-formed schema and found a violation.
    TECHNICAL  validity could not be established (unreachable / slow / malformed external reference, malformed schema, any
               other exception). It is never a proof of invalidity and never a pass: callers must fail closed.

Remote `$ref` documents are fetched here, not by jsonschema (whose own retrieval has no timeout and is deprecated): bounded
timeout, bounded retries for transient failures only, a size limit, only http(s), and an in-process cache so that one process
validates against one fixed copy of each document. Nothing is written to disk. A document that failed all its attempts is
remembered for a short time so an outage cannot stall hundreds of validations for the full retry time each.
"""
import hashlib
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import jsonschema
from jsonschema import validators

VALID, INVALID, TECHNICAL = "valid", "invalid", "technical"

REMOTE_TIMEOUT_SECONDS = 10          # per socket operation (connect, each read): it does NOT bound the whole download
REMOTE_TOTAL_SECONDS = 30            # per attempt, wall clock, headers and body included (a slowly dripping server cannot stall us)
REMOTE_ATTEMPTS = 3                  # total tries of one document, transient failures only
REMOTE_BACKOFF_SECONDS = (1.0, 3.0)  # waits before the 2nd and 3rd try
FAILURE_MEMO_SECONDS = 120           # a document that failed all its attempts is not tried again for this long
MAX_REMOTE_BYTES = 8 * 1024 * 1024
MAX_CACHED_DOCUMENTS = 256
MAX_CACHED_BYTES = 64 * 1024 * 1024  # raw size of all cached documents together
MAX_STALLED_WORKERS = 8              # downloads we gave up on that are still running; above this no new download is started
MAX_REMEMBERED_FAILURES = 1024

_sleep, _clock = time.sleep, time.monotonic          # replaced by the tests
_opener = None                                       # tests may install a callable(request, timeout) -> response
_documents = {}                                      # uri -> parsed document (successful fetches only)
_digests = {}                                        # uri -> sha256 of the bytes that were parsed (report only, never persisted)
_sizes = {}                                          # uri -> raw size of the cached document
_workers = set()                                     # download threads that have started and not finished
_workers_lock = threading.Lock()
_local = threading.local()                           # the last remote-reference failure seen by this thread's validation
_failures = {}                                       # uri -> (clock time, message) of a document that exhausted its attempts


class RemoteRefError(Exception):
    """A remote `$ref` document could not be obtained. `retriable` tells the retry loop whether another try can help."""

    def __init__(self, kind, message, retriable=False):
        super().__init__(f"{kind}: {message}")
        self.kind, self.retriable = kind, retriable


def reset_caches():
    _documents.clear()
    _digests.clear()
    _sizes.clear()
    _failures.clear()


def _display(uri):
    """The uri without any user:password part, for reports."""
    parts = urllib.parse.urlsplit(uri)
    host = parts.netloc.rpartition("@")[2]
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, parts.query, ""))


def used_documents():
    """{uri: sha256 of the fetched bytes} for every remote document this process validated against (for run reports only:
    read-only information, it influences no verdict and is written nowhere)."""
    return dict(sorted((_display(u), d) for u, d in _digests.items()))


def stalled_workers():
    """Number of download threads still running (after a healthy attempt this is 0)."""
    with _workers_lock:
        return len(_workers)


def _open(request):
    if _opener is not None:
        return _opener(request, REMOTE_TIMEOUT_SECONDS)
    host = urllib.parse.urlparse(request.full_url).hostname
    loopback = host in ("127.0.0.1", "localhost", "::1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if loopback else urllib.request.build_opener()
    return opener.open(request, timeout=REMOTE_TIMEOUT_SECONDS)


def _fetch_blocking(uri, deadline):
    """One download. Returns (document, sha256, size). Every read is checked against `deadline`, so a server that keeps sending
    a few bytes at a time cannot hold the body open past it (the socket timeout alone only bounds each single operation)."""
    request = urllib.request.Request(uri, headers={"Accept": "application/json", "User-Agent": "benchson-schema-check"})
    try:
        with _open(request) as response:
            chunks, size = [], 0
            read = getattr(response, "read1", None) or response.read     # read1 returns after ONE network read: read(n) would
            while True:                                                  # keep collecting n bytes and never reach the deadline check
                chunk = read(8192)
                if not chunk:
                    break
                chunks.append(chunk); size += len(chunk)
                if size > MAX_REMOTE_BYTES:
                    raise RemoteRefError("too_large", f"{uri} exceeds {MAX_REMOTE_BYTES} bytes")
                if _clock() > deadline:
                    raise RemoteRefError("timeout", f"{uri} (total time limit {REMOTE_TOTAL_SECONDS}s)", retriable=True)
    except RemoteRefError:
        raise
    except urllib.error.HTTPError as e:
        # 408 / 429 / 5xx are transient; any other status is the server's definite answer
        raise RemoteRefError(f"http_{e.code}", uri, retriable=e.code in (408, 429) or e.code >= 500)
    except (TimeoutError, socket.timeout) as e:
        raise RemoteRefError("timeout", f"{uri} ({type(e).__name__})", retriable=True)
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", None)
        kind = "timeout" if isinstance(reason, (TimeoutError, socket.timeout)) else "connection"
        raise RemoteRefError(kind, f"{uri} ({type(reason).__name__ if reason else type(e).__name__})", retriable=True)
    except OSError as e:
        raise RemoteRefError("connection", f"{uri} ({type(e).__name__})", retriable=True)
    raw = b"".join(chunks)
    try:
        return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), len(raw)
    except ValueError as e:
        raise RemoteRefError("invalid_json", f"{uri} ({type(e).__name__})")


def _fetch_once(uri):
    """One attempt with a hard wall-clock limit (REMOTE_TOTAL_SECONDS) that also covers connecting and the response headers.
    The download runs in a daemon thread; if the limit passes the attempt is a timeout and the thread is left to finish on its
    own (its socket operations time out, its reads check the same deadline, and it writes nothing anywhere)."""
    outcome = []
    deadline = _clock() + REMOTE_TOTAL_SECONDS
    with _workers_lock:
        if len(_workers) >= MAX_STALLED_WORKERS:     # earlier downloads are still hanging: do not pile up more threads or sockets
            raise RemoteRefError("too_many_stalled_requests", f"{uri} ({len(_workers)} earlier downloads have not finished)")

    def work():
        try:
            outcome.append(("ok", _fetch_blocking(uri, deadline)))
        except BaseException as e:           # reported to the caller, never lost
            outcome.append(("error", e))
        finally:
            with _workers_lock:
                _workers.discard(threading.current_thread())
    worker = threading.Thread(target=work, daemon=True, name="benchson-remote-ref")
    with _workers_lock:
        _workers.add(worker)
    worker.start()
    worker.join(REMOTE_TOTAL_SECONDS)
    if not outcome:
        raise RemoteRefError("timeout", f"{uri} (no complete answer within {REMOTE_TOTAL_SECONDS}s)", retriable=True)
    kind, value = outcome[0]
    if kind == "error":
        if isinstance(value, RemoteRefError):
            raise value
        raise RemoteRefError("internal", f"{uri} ({type(value).__name__})")      # not a network problem: retrying cannot help
    return value


def _remember_document(uri, document, digest, size):
    """In-process cache bounded by count and by total raw size (the oldest documents go first)."""
    while _documents and (len(_documents) >= MAX_CACHED_DOCUMENTS or sum(_sizes.values()) + size > MAX_CACHED_BYTES):
        oldest = next(iter(_documents))
        _documents.pop(oldest); _digests.pop(oldest, None); _sizes.pop(oldest, None)
    _documents[uri], _digests[uri], _sizes[uri] = document, digest, size


def fetch_remote_document(uri):
    """The parsed JSON document at `uri`; raises RemoteRefError. Bounded: at most REMOTE_ATTEMPTS tries, one cache per process."""
    try:
        return _fetch_remote_document(uri)
    except RemoteRefError as e:
        _local.last = e          # jsonschema 4.x wraps the cause away; keep it for the report
        raise


def _fetch_remote_document(uri):
    if urllib.parse.urlparse(uri).scheme not in ("http", "https"):
        raise RemoteRefError("unsupported_scheme", uri)
    if uri in _documents:
        return _documents[uri]
    failed = _failures.get(uri)
    if failed and _clock() - failed[0] < FAILURE_MEMO_SECONDS:
        raise RemoteRefError("recent_failure", f"{failed[1]} (not retried for {FAILURE_MEMO_SECONDS}s)")
    last = None
    for attempt in range(1, REMOTE_ATTEMPTS + 1):
        try:
            document, digest, size = _fetch_once(uri)
        except RemoteRefError as e:
            last = e
            if not e.retriable or attempt == REMOTE_ATTEMPTS:
                break
            _sleep(REMOTE_BACKOFF_SECONDS[min(attempt - 1, len(REMOTE_BACKOFF_SECONDS) - 1)])
            continue
        _remember_document(uri, document, digest, size)
        _failures.pop(uri, None)
        return document
    if len(_failures) >= MAX_REMEMBERED_FAILURES:
        now = _clock()
        for old in [u for u, (t, _) in _failures.items() if now - t >= FAILURE_MEMO_SECONDS] or [next(iter(_failures))]:
            _failures.pop(old, None)
    _failures[uri] = (_clock(), str(last))
    raise last


def _validator(schema):
    """The validator class `jsonschema.validate` would pick, wired to our bounded retrieval (jsonschema 4.x and 3.x)."""
    cls = validators.validator_for(schema)
    cls.check_schema(schema)
    try:
        import referencing
        import referencing.jsonschema
        from jsonschema_specifications import REGISTRY
    except ImportError:                                   # jsonschema < 4.18: RefResolver with our handler for http(s)
        handler = fetch_remote_document
        resolver = validators.RefResolver.from_schema(schema, id_of=cls.ID_OF, handlers={"http": handler, "https": handler})
        return cls(schema, resolver=resolver)

    def retrieve(uri):
        return referencing.Resource.from_contents(fetch_remote_document(uri),
                                                  default_specification=referencing.jsonschema.DRAFT202012)
    return cls(schema, registry=REGISTRY.combine(referencing.Registry(retrieve=retrieve)))


def _describe(error):
    text = f"{type(error).__name__}: {str(error)[:200]}"
    cause = error.__cause__ or error.__context__
    return text + (f" <- {type(cause).__name__}: {str(cause)[:200]}" if cause is not None else "")


def validate_instance(obj, schema):
    """(status, detail): VALID / INVALID / TECHNICAL. `detail` is "valid", the first violation message, or the technical cause.

    All errors are collected: if ANY part of the validation fails technically the whole result is TECHNICAL, even when a
    violation was already seen, so the outcome does not depend on the order in which the validator happens to visit keywords."""
    _local.last = None
    try:
        validator = _validator(schema)
        errors = list(validator.iter_errors(obj))
    except jsonschema.SchemaError as e:
        return TECHNICAL, f"malformed schema: {str(e.message)[:200]}"
    except Exception as e:           # unreachable / slow / malformed $ref, recursion, anything else: NOT evidence about obj
        remote = getattr(_local, "last", None)
        return TECHNICAL, (f"external reference could not be retrieved: {str(remote)[:300]}" if remote is not None else _describe(e))
    if errors:
        return INVALID, jsonschema.exceptions.best_match(errors).message
    return VALID, "valid"
