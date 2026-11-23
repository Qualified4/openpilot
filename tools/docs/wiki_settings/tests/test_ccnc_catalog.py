"""CCNC catalog integration checks, separate from the original Wiki tests."""
import json
from pathlib import Path
import tempfile
import unittest

from test_generate import GENERATOR, VALIDATOR, COMMIT, STAMP
from test_ci_check import CI_CHECK


CATALOG = Path(__file__).resolve().parents[4] / "openpilot/selfdrive/carrot_settings.json"


class CcncCatalogTest(unittest.TestCase):
  def test_current_catalog_and_localized_descriptions(self):
    settings, _ = GENERATOR.load_catalog(CATALOG, ("ko", "en", "zh"))
    result = GENERATOR.generate(CATALOG, catalog_commit=COMMIT, generated_at=STAMP)
    self.assertEqual(set(result.index["settings"]), {setting.param for setting in settings})
    self.assertEqual(len(result.pages), len(settings) * 3 + 2)
    params = {item["name"]: item for item in json.loads(CATALOG.read_text(encoding="utf-8"))["params"]}
    self.assertEqual({key for key in params if key.startswith("Ccnc")},
                     {"CcncClusterDisplay", "CcncExtendedDisplay", "CcncMapProcess", "CcncVisionProcess",
                      "CcncTrafficBackground"})
    key = "CcncExtendedDisplay"
    setting = params[key]
    self.assertEqual((setting["control"], setting["min"], setting["max"], setting["default"]), ("toggle", 0, 1, 0))
    self.assertTrue(setting["requires_ccnc_hda1"])
    self.assertEqual(setting["title"], "CCNC 확장 표시 (HDA1)")
    self.assertEqual((params["CcncClusterDisplay"]["default"], params["CcncTrafficBackground"]["default"]), (0, 1))
    for locale, field in (("ko", "descr"), ("en", "edescr"), ("zh", "cdescr")):
      page = result.pages[result.index["settings"][key]["locales"][locale]["page"]]
      self.assertEqual(VALIDATOR.validate_wiki_markdown(page), [])
      self.assertIn(" ".join(setting[field].split()), page)
      self.assertEqual(setting[field].count("\n• "), 8)
      self.assertNotIn("TBD", setting[field])
      for value in ("0.9", "2.4", "3.1"):
        self.assertIn(value, setting[field])
      self.assertIn("14.7", setting[field])

  def test_current_catalog_ci_report(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      wiki = root / "wiki"
      wiki.mkdir()
      status, report = CI_CHECK.run_check(
        catalog=CATALOG, wiki_dir=wiki, output_dir=root / "report",
        locales=("ko", "en", "zh"), catalog_commit=COMMIT, wiki_commit=None,
      )
      self.assertEqual(status, 0)
      self.assertEqual(report["validationIssues"], [])
      stored = json.loads((root / "report/summary.json").read_text(encoding="utf-8"))
      settings, _ = GENERATOR.load_catalog(CATALOG)
      self.assertEqual(stored["result"]["settings"], len(settings))
      self.assertEqual(list(wiki.iterdir()), [])


if __name__ == "__main__":
  unittest.main()
