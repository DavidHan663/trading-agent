"""Prevent empty template values from hiding adjacent real credentials."""

from scripts.check_secrets import findings


def test_empty_values_and_documentation_placeholders_are_not_secrets(tmp_path):
    path = tmp_path / "example.env"
    path.write_text('OPENAI_API_KEY=\nGOOGLE_API_KEY=\nANTHROPIC_API_KEY=your_key\nPASSWORD=""\n')
    assert findings(path) == []


def test_scan_reports_locations_without_values(tmp_path):
    path = tmp_path / "example.env"
    path.write_text('OPENAI_API_KEY=\nCUSTOM_API_KEY=not-a-real-credential-but-populated\n')
    assert findings(path) == [(2, "populated env secret")]
