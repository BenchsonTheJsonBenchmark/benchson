"""Structure-only inspection of a PEM-like private-key block. Standard library only.

Used by `hf_baseline.py inspect-credential`. Everything returned is safe metadata (format names, lengths, booleans,
public algorithm identifiers). The key text and the decoded bytes are never returned, printed or logged.
This module only describes; it never decides that something is safe to publish.
"""
import base64
import binascii
import hashlib
import math
import re
import textwrap

KNOWN_OIDS = {"1.2.840.113549.1.1.1": "rsaEncryption", "1.2.840.10045.2.1": "ecPublicKey", "1.3.101.112": "Ed25519",
              "1.3.101.113": "Ed448", "1.3.101.110": "X25519", "1.2.840.10040.4.1": "dsa",
              "1.2.840.113549.1.5.13": "pbes2", "1.2.840.10045.3.1.7": "prime256v1", "1.3.132.0.34": "secp384r1",
              "1.3.132.0.35": "secp521r1", "1.3.132.0.10": "secp256k1"}
TAG_NAMES = {0x02: "INTEGER", 0x03: "BIT STRING", 0x04: "OCTET STRING", 0x05: "NULL", 0x0C: "UTF8String",
             0x30: "SEQUENCE", 0x31: "SET"}
OPENSSH_MAGIC = b"openssh-key-v1\x00"


def _read(buf, off):
    """DER header at `off` -> (tag, content_length, content_start). ValueError if malformed."""
    if off + 2 > len(buf):
        raise ValueError("truncated header")
    tag, first = buf[off], buf[off + 1]
    off += 2
    if first < 0x80:
        return tag, first, off
    n = first & 0x7F
    if n == 0 or n > 4 or off + n > len(buf):
        raise ValueError("bad length")
    return tag, int.from_bytes(buf[off:off + n], "big"), off + n


def _children(buf, start, end):
    out, off = [], start
    while off < end:
        tag, length, cs = _read(buf, off)
        if cs + length > end:
            raise ValueError("child overruns its parent")
        out.append((tag, length, cs))
        off = cs + length
    return out


def _oid(content):
    if not content:
        return "?"
    a = min(content[0] // 40, 2)
    parts, v = [a, content[0] - 40 * a], 0
    for x in content[1:]:
        v = (v << 7) | (x & 0x7F)
        if not x & 0x80:
            parts.append(v)
            v = 0
    dotted = ".".join(map(str, parts))
    return KNOWN_OIDS.get(dotted, dotted)


def _describe(buf, tag, length, cs, depth):
    if tag == 0x06:
        return "OID " + _oid(buf[cs:cs + length])
    if tag == 0x30 and depth > 0:
        return "SEQUENCE[" + ", ".join(_describe(buf, *k, depth - 1) for k in _children(buf, cs, cs + length)) + "]"
    if 0xA0 <= tag <= 0xBF:
        return f"[{tag & 0x1F}]({length}B)"
    return f"{TAG_NAMES.get(tag, f'tag 0x{tag:02x}')}({length}B)"


def _entropy_bytes(data):
    n = len(data)
    counts = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    return -sum(c / n * math.log2(c / n) for c in counts.values()) if n else 0.0


def _periodic(data, max_period=32):
    for p in range(1, min(max_period, len(data) // 2) + 1):
        if data == (data[:p] * (len(data) // p + 1))[:len(data)]:
            return p
    return 0


def _try_cryptography(label, body):
    """Optional second opinion if the `cryptography` package is installed. The key object is never kept or printed."""
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        return "cryptography package not installed (optional check skipped)"
    pem = ("-----BEGIN %s-----\n%s\n-----END %s-----\n" % (label, "\n".join(textwrap.wrap(body, 64)), label)).encode()
    try:
        key = serialization.load_pem_private_key(pem, password=None)
    except TypeError:
        return "parsed as an encrypted private key (password required)"
    except Exception as e:  # noqa: BLE001 - only the class name is reported
        return f"not parseable as a private key ({type(e).__name__})"
    size = getattr(key, "key_size", None)
    return f"PARSED as a valid private key: {type(key).__name__}" + (f", {size} bits" if size else "")


def _extract(match, full, start):
    """Split the block that starts with header `match` into its parts (internal; carries the value)."""
    label = match[len("-----BEGIN "):-len("-----")]
    rest = full[start + len(match):]
    e = rest.find("-----END")
    footer = re.match(r"-----END ([A-Z ]*)-----", rest[e:]) if e >= 0 else None
    raw = rest[:e] if e >= 0 else rest
    norm = re.sub(r"\\[nr]", "\n", raw)
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]
    body_lines = [ln for ln in lines if ":" not in ln]
    return {"label": label, "e": e, "footer": footer, "raw": raw, "lines": lines, "body_lines": body_lines,
            "body": "".join(body_lines)}


def body_sha256(match, full, start):
    """Full SHA-256 (hex) of the normalized encoded body (whitespace, escaped newlines and PEM header lines removed)."""
    return hashlib.sha256(_extract(match, full, start)["body"].encode("utf-8", "ignore")).hexdigest()


def rsa_prefix_report(buf):
    """Lenient walk over a possibly truncated RSA private-key container (PKCS#8 or PKCS#1).

    Returns safe facts only (None when the bytes do not start like an RSA key container). The key point is
    `bytes_after_modulus`: everything after the modulus is private (exponent, primes, CRT values); the modulus itself is public.
    """
    def rd(off):
        try:
            return _read(buf, off)
        except ValueError:
            return None

    def fits(item):  # the header and the whole content are inside the buffer
        return item is not None and item[2] + item[1] <= len(buf)

    top = rd(0)
    if top is None or top[0] != 0x30:
        return None
    ver = rd(top[2])
    if ver is None or ver[0] != 2 or ver[1] != 1 or ver[2] >= len(buf) or buf[ver[2]] != 0:
        return None
    nxt = rd(ver[2] + 1)
    layout, mod = None, None
    if nxt is not None and nxt[0] == 0x30:  # PKCS#8: AlgorithmIdentifier, then OCTET STRING { RSAPrivateKey }
        oid = rd(nxt[2])
        if oid is None or oid[0] != 6 or not fits(oid) or _oid(buf[oid[2]:oid[2] + oid[1]]) != "rsaEncryption":
            return None
        octets = rd(nxt[2] + nxt[1])
        inner = rd(octets[2]) if octets is not None and octets[0] == 4 else None
        iver = rd(inner[2]) if inner is not None and inner[0] == 0x30 else None
        if iver is None or iver[0] != 2 or iver[1] != 1 or iver[2] >= len(buf) or buf[iver[2]] != 0:
            return None
        mod, layout = rd(iver[2] + 1), "PKCS#8 RSA"
    elif nxt is not None and nxt[0] == 2:  # PKCS#1: INTEGER 0, then the modulus
        mod, layout = nxt, "PKCS#1 RSA"
    if mod is None or mod[0] != 2 or layout is None:
        return None
    mod_end = mod[2] + mod[1]
    return {"layout": layout, "declared_total_bytes": top[2] + top[1], "actual_bytes": len(buf),
            "modulus_declared_bytes": mod[1], "modulus_bytes_present": max(0, min(len(buf), mod_end) - mod[2]),
            "bytes_after_modulus": max(0, len(buf) - mod_end)}


def inspect_pem_like(match, full, start):
    """Safe metadata about the block that starts with the header `match` at `full[start:]`."""
    x = _extract(match, full, start)
    label, e, footer, raw, lines, body_lines, body = (x[k] for k in ("label", "e", "footer", "raw", "lines", "body_lines", "body"))
    pem_header_lines = [ln for ln in lines if ":" in ln]
    info = {
        "label": label, "footer_present": e >= 0, "footer_well_formed": bool(footer),
        "footer_label_matches_header": bool(footer) and footer.group(1) == label,
        "escaped_newlines": len(re.findall(r"\\[nr]", raw)), "real_newlines": raw.count("\n"),
        "pem_header_lines": len(pem_header_lines), "body_chars": len(body), "body_lines": len(body_lines),
        "line_lengths": sorted({len(ln) for ln in body_lines})[:4],
        "body_fingerprint": hashlib.sha256(body.encode("utf-8", "ignore")).hexdigest()[:8],
        "body_sha256": hashlib.sha256(body.encode("utf-8", "ignore")).hexdigest(),
    }
    non_b64 = re.sub(r"[A-Za-z0-9+/=_\-]", "", body)
    info["chars_outside_base64_alphabets"] = len(non_b64)
    decoded = None
    if body and not non_b64:
        urlsafe = "-" in body or "_" in body
        padded = body + "=" * (-len(body.rstrip("=")) % 4) if "=" not in body else body
        try:
            decoded = (base64.urlsafe_b64decode if urlsafe else base64.b64decode)(padded)
            info["base64"] = ("url-safe" if urlsafe else "standard") + ("" if "=" in body or len(body) % 4 == 0 else ", padding added")
        except (binascii.Error, ValueError):
            decoded = None
    if decoded is None:
        info["base64"] = "invalid or not base64"
        info["decoded_bytes"] = None
        info["der"] = None
        info["cryptography"] = "skipped (body is not base64)"
        info["format"] = "none"
        info["verdict"], info["reason"] = "synthetic_non_key_body", "the body is not a base64 encoding at all"
        return info

    info["decoded_bytes"] = len(decoded)
    info["decoded_entropy_bits_per_byte"] = round(_entropy_bytes(decoded), 2)
    info["decoded_printable_ascii_fraction"] = round(sum(32 <= b < 127 or b in (9, 10, 13) for b in decoded) / len(decoded), 2)
    info["decoded_periodic_pattern_length"] = _periodic(decoded)
    info["openssh_magic"] = decoded.startswith(OPENSSH_MAGIC)
    der = {"top_level_asn1_sequence": False}
    fmt = "none"
    try:
        tag, length, cs = _read(decoded, 0)
        total = cs + length
        der.update(top_tag=TAG_NAMES.get(tag, f"0x{tag:02x}"), top_level_asn1_sequence=(tag == 0x30),
                   declared_total_bytes=total, actual_bytes=len(decoded), complete=(total == len(decoded)),
                   missing_bytes=max(0, total - len(decoded)), trailing_bytes=max(0, len(decoded) - total))
        if tag == 0x30:
            end = min(total, len(decoded))
            try:
                kids = _children(decoded, cs, end)
                der["outline"] = _describe(decoded, tag, length, cs, 2) if total <= len(decoded) else "(truncated)"
                tags = [k[0] for k in kids]
                if len(kids) >= 9 and tags[:9] == [2] * 9 and kids[0][1] == 1 and decoded[kids[0][2]] == 0:
                    fmt = "RSA private key (PKCS#1)"
                    mod = decoded[kids[1][2]:kids[1][2] + kids[1][1]]
                    der["rsa_modulus_bits"] = (len(mod) - (1 if mod[:1] == b"\x00" else 0)) * 8
                elif len(kids) >= 3 and tags[0] == 2 and tags[1] == 0x30 and tags[2] == 4:
                    alg = _children(decoded, kids[1][2], kids[1][2] + kids[1][1])
                    algname = _oid(decoded[alg[0][2]:alg[0][2] + alg[0][1]]) if alg and alg[0][0] == 6 else "?"
                    fmt = f"PKCS#8 private key ({algname})"
                    der["inner_key_octets"] = kids[2][1]
                elif len(kids) >= 2 and tags[0] == 2 and kids[0][1] == 1 and decoded[kids[0][2]] == 1 and tags[1] == 4:
                    fmt = "EC private key (SEC1)"
                    der["inner_key_octets"] = kids[1][1]
                elif len(kids) == 2 and tags == [0x30, 0x04]:
                    fmt = "encrypted PKCS#8 private key"
            except ValueError:
                der["outline"] = "malformed inside the sequence"
    except ValueError:
        der["header"] = "not a valid DER header"
    info["der"] = der
    rsa = rsa_prefix_report(decoded) if der.get("top_level_asn1_sequence") else None
    info["rsa_prefix"] = rsa
    truncated_rsa = bool(rsa) and not der.get("complete") and rsa["bytes_after_modulus"] == 0 \
        and rsa["modulus_bytes_present"] < rsa["modulus_declared_bytes"] and not info["openssh_magic"]
    if truncated_rsa and fmt == "none":
        fmt = f"truncated {rsa['layout']} key container"
    info["format"] = "OpenSSH private key container" if info["openssh_magic"] else fmt
    info["cryptography"] = _try_cryptography(label, body)

    parsed = info["cryptography"].startswith(("PARSED", "parsed as an encrypted"))
    if parsed or (der.get("complete") and fmt != "none"):
        info["verdict"], info["reason"] = "complete_parseable_private_key", "structure matches a complete private key; treat as a real, exposed key"
    elif truncated_rsa:
        info["verdict"] = "truncated_rsa_key_prefix_public_part_only"
        info["reason"] = ("the first bytes of an RSA key container, cut off inside the modulus: it holds no private exponent, "
                          "primes or CRT values (only part of the public modulus) and cannot be parsed or completed into a key")
    elif info["openssh_magic"] or (der.get("top_level_asn1_sequence") and fmt != "none"):
        info["verdict"], info["reason"] = "partial_private_key_structure", "real key format but incomplete or unverifiable; may leak key material"
    elif der.get("top_level_asn1_sequence"):
        info["verdict"], info["reason"] = "unrecognized_der_structure", "valid DER sequence of an unknown kind; cannot rule out a key"
    else:
        info["verdict"], info["reason"] = "synthetic_non_key_body", "base64 data that is not DER and not an OpenSSH container: not a usable private key in any standard format"
    return info
