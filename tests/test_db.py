"""DB tests: idempotent schema, blob round-trip, model cache."""

from claimlens import db


def test_init_schema_twice_and_blob_round_trip():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    db.init_schema(conn)  # must not error
    blob = bytes(range(256)) * 4
    conn.execute("INSERT INTO cars (car_id) VALUES ('c1')")
    conn.execute(
        "INSERT INTO photos (photo_id, car_id, photo_idx, width, height, img)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ["p1", "c1", 0, 10, 10, blob],
    )
    (back,) = conn.execute(
        "SELECT img FROM photos WHERE photo_id = 'p1'"
    ).fetchone()
    assert bytes(back) == blob
    conn.close()


def test_model_cache_put_get_and_overwrite():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    assert db.cache_get(conn, "nope") is None
    db.cache_put(conn, "k1", {"a": 1})
    assert db.cache_get(conn, "k1") == {"a": 1}
    db.cache_put(conn, "k1", {"a": 2})
    assert db.cache_get(conn, "k1") == {"a": 2}
    conn.close()
