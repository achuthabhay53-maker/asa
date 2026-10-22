from aca.evidence import canonicalize_evidence_url


def test_canonicalize_evidence_url_normalizes_host_scheme_port_fragment_and_slash():
    assert (
        canonicalize_evidence_url("HTTPS://WWW.Example.com:443/products/overview/#specs")
        == "https://www.example.com/products/overview"
    )


def test_canonicalize_evidence_url_preserves_meaningful_path_and_query():
    left = canonicalize_evidence_url("https://example.com/a/?id=1")
    right = canonicalize_evidence_url("https://example.com/b?id=1")
    other_query = canonicalize_evidence_url("https://example.com/a?id=2")

    assert left == "https://example.com/a?id=1"
    assert left != right
    assert left != other_query


def test_canonicalize_evidence_url_rejects_invalid_and_credential_urls():
    assert canonicalize_evidence_url("not-a-url") == ""
    assert canonicalize_evidence_url("https://user:secret@example.com/page") == ""
