#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import real_consultation_keyword_evaluate as ev


class RealConsultationKeywordEvaluationTests(unittest.TestCase):
    @staticmethod
    def generated_definition():
        keywords = []
        for index in range(10):
            keywords.append({
                "keyword": f"概念{index}",
                "category": "其他",
                "accepted_forms": [
                    f"概念{index}", f"表達{index}", f"term {index}"
                ],
                "doctor_evidence": f"evidence {index}",
            })
        return {
            "title": "Test consultation",
            "keywords": keywords,
        }, " ".join(f"evidence {index}" for index in range(10))

    def test_normalization_is_format_only(self):
        self.assertEqual(ev.normalize_text(" COVID／Flu RAT! "), "covidflurat")
        self.assertNotEqual(ev.normalize_text("冇"), ev.normalize_text("無"))

    def test_keyword_match_accepts_configured_form(self):
        keyword = ev.Keyword("流感", "診斷", ("流行性感冒",), "flu")
        self.assertEqual(ev.keyword_match(keyword, "找到流行性感冒。"), "流行性感冒")
        self.assertEqual(ev.keyword_match(keyword, "普通感冒。"), "")

    def test_keyword_match_accepts_balanced_medicine_forms(self):
        keyword = ev.Keyword(
            "撲熱息痛", "藥物",
            ("Paracetamol", "Panadol", "止痛藥", "退燒藥"),
            "PARACETAMOL TAB 500MG",
        )
        for transcript, expected in (
            ("醫生開咗 Paracetamol。", "Paracetamol"),
            ("我食咗撲熱息痛。", "撲熱息痛"),
            ("會開止痛藥俾你。", "止痛藥"),
        ):
            with self.subTest(transcript=transcript):
                self.assertEqual(ev.keyword_match(keyword, transcript), expected)
        self.assertEqual(ev.keyword_match(keyword, "醫生開咗抗生素。"), "")

    def test_generated_keywords_require_at_least_three_forms(self):
        definition, appointment = self.generated_definition()
        definition["keywords"][0]["accepted_forms"] = ["概念0", "term 0"]
        with self.assertRaisesRegex(ValueError, "requires 3-8 accepted forms"):
            ev.validate_generated_definition("sample", definition, appointment)

    def test_generated_keywords_reject_normalized_duplicate_forms(self):
        definition, appointment = self.generated_definition()
        definition["keywords"][0]["accepted_forms"] = [
            "概念0", "Panadol", "Ｐａｎａｄｏｌ"
        ]
        with self.assertRaisesRegex(ValueError, "duplicate accepted form"):
            ev.validate_generated_definition("sample", definition, appointment)

    def test_generated_keywords_reject_forms_shared_across_groups(self):
        definition, appointment = self.generated_definition()
        definition["keywords"][0]["accepted_forms"][1] = "共同表達"
        definition["keywords"][1]["accepted_forms"][1] = "共同表達"
        with self.assertRaisesRegex(ValueError, "shared by keywords"):
            ev.validate_generated_definition("sample", definition, appointment)

    def test_negative_keyword_requires_negation_in_every_form(self):
        definition, appointment = self.generated_definition()
        definition["keywords"][0].update({
            "keyword": "冇發燒",
            "accepted_forms": ["冇發燒", "無發燒", "fever"],
        })
        with self.assertRaisesRegex(ValueError, "forms without negation"):
            ev.validate_generated_definition("sample", definition, appointment)

    def test_medical_abbreviation_can_preserve_negation(self):
        definition, appointment = self.generated_definition()
        definition["keywords"][0].update({
            "keyword": "冇藥物敏感",
            "accepted_forms": ["冇藥物敏感", "無藥物敏感", "NKDA"],
        })
        cleaned = ev.validate_generated_definition(
            "sample", definition, appointment
        )
        self.assertEqual(
            cleaned["keywords"][0]["accepted_forms"][-1], "NKDA"
        )

    def test_generated_keywords_reject_mandarin_only_forms(self):
        definition, appointment = self.generated_definition()
        definition["keywords"][0]["accepted_forms"][1] = "這兩日"
        with self.assertRaisesRegex(ValueError, "non-Hong-Kong-Cantonese"):
            ev.validate_generated_definition("sample", definition, appointment)

    def test_generated_keywords_separate_medicine_and_dose(self):
        definition, appointment = self.generated_definition()
        definition["keywords"][0].update({
            "keyword": "撲熱息痛",
            "accepted_forms": ["撲熱息痛", "Paracetamol", "Panadol"],
        })
        definition["keywords"][1].update({
            "keyword": "撲熱息痛五百毫克",
            "accepted_forms": [
                "撲熱息痛五百毫克", "Paracetamol 500mg", "Panadol 500mg"
            ],
        })
        with self.assertRaisesRegex(ValueError, "Medication and dose must be separate"):
            ev.validate_generated_definition("sample", definition, appointment)

    def test_flatten_json_text_removes_html(self):
        value = [{"diagnosis_detail": {"History": "<p>No <b>fever</b></p>"}}]
        self.assertEqual(ev.flatten_json_text(value), "No fever")

    def test_plain_text_appointment_and_recording_are_discovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            note = root / "sample.appointment.txt"
            audio = root / "sample.wav"
            note.write_text("Sore throat. No fever.", encoding="utf-8")
            audio.write_bytes(b"RIFF-not-real-audio")
            pairs = ev.discover_input_pairs(root)
        self.assertEqual(pairs["sample"], (note, audio))

    def test_input_discovery_rejects_unpaired_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "orphan.wav").write_bytes(b"RIFF-not-real-audio")
            with self.assertRaisesRegex(ValueError, "no matching appointment note"):
                ev.discover_input_pairs(root)

    def test_input_discovery_rejects_duplicate_appointment_notes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.appointment.txt").write_text(
                "Sore throat", encoding="utf-8"
            )
            (root / "sample.record_appointment.json").write_text(
                json.dumps({"History": "Sore throat"}), encoding="utf-8"
            )
            (root / "sample.wav").write_bytes(b"RIFF-not-real-audio")
            with self.assertRaisesRegex(ValueError, "Multiple appointment notes"):
                ev.discover_input_pairs(root)

    def test_environment_builds_zero_argument_pipeline_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                "INPUT_DIR": "incoming",
                "OUTPUT_DIR": "artifacts",
                "KEYWORD_MODEL": "gemini-test",
                "ASR_MODELS": "qwen3-asr-flash",
                "GEMINI_API_KEY": "gemini-secret",
                "DASHSCOPE_API_KEY": "dashscope-secret",
            }
            with mock.patch.dict("os.environ", environment, clear=True):
                config = ev.build_pipeline_config(root)
        root = root.resolve()
        self.assertEqual(config.input_dir, root / "incoming")
        self.assertEqual(config.output_dir, root / "artifacts")
        self.assertEqual(
            config.keyword_config, root / "artifacts/generated_keywords.json"
        )
        self.assertEqual(config.keyword_model, "gemini-test")
        self.assertEqual(
            [model.model_id for model in config.asr_models],
            ["qwen3-asr-flash"],
        )

    def test_pipeline_interface_writes_report_and_safe_run_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "results"
            transcript_dir = root / "existing-transcripts"
            input_dir.mkdir()
            transcript_dir.mkdir()
            (input_dir / "sample.appointment.txt").write_text(
                "Sore throat. No fever.", encoding="utf-8"
            )
            (input_dir / "sample.wav").write_bytes(b"RIFF-not-real-audio")
            (transcript_dir / "sample.txt").write_text(
                "The patient has a sore throat.", encoding="utf-8"
            )
            keyword_config = root / "keywords.csv"
            keyword_config.write_text(
                "consultation_id,diagnosis,keyword,category,accepted_forms,"
                "doctor_evidence\n"
                "sample,URTI,喉嚨痛,症狀,喉嚨痛 | sore throat,Sore throat\n",
                encoding="utf-8-sig",
            )
            config = ev.PipelineConfig(
                root=root,
                input_dir=input_dir,
                output_dir=output_dir,
                keyword_config=keyword_config,
                keyword_model="unused-test-model",
                asr_models=(ev.ASR_MODELS["qwen3-asr-flash"],),
                gemini_api_key="must-not-be-written",
                dashscope_api_key="must-not-be-written",
            )
            report = ev.run_pipeline(
                config, model_outputs={transcript_dir: "existing-asr"}
            )
            manifest_text = (output_dir / "run_manifest.json").read_text()
            summary_text = (output_dir / "summary.csv").read_text(
                encoding="utf-8-sig"
            )
        self.assertEqual(report, output_dir / "ASR-evaluation-report.md")
        self.assertIn("existing-asr", summary_text)
        self.assertIn('"status": "completed"', manifest_text)
        self.assertNotIn("must-not-be-written", manifest_text)

    def test_generated_keyword_config_is_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "sample.record_appointment.json"
            audio = root / "sample.wav"
            config = root / "generated_keywords.json"
            record.write_text(
                json.dumps([{"diagnosis": "URTI", "History": "No fever"}]),
                encoding="utf-8",
            )
            audio.write_bytes(b"RIFF-not-real-audio")
            config.write_text(json.dumps({
                "schema_version": 1,
                "consultations": {
                    "sample": {
                        "title": "URTI",
                        "keywords": [{
                            "keyword": "fever",
                            "category": "症狀",
                            "accepted_forms": ["發燒"],
                            "doctor_evidence": "No fever",
                        }],
                    }
                },
            }), encoding="utf-8")
            consultations = ev.load_consultations(root, config)
        self.assertEqual([item.consultation_id for item in consultations], ["sample"])
        self.assertEqual(consultations[0].keywords[0].accepted_forms, ("發燒",))

    def test_keyword_generation_accumulates_validation_feedback(self):
        valid, appointment = self.generated_definition()
        first = json.loads(json.dumps(valid))
        first["keywords"][0]["accepted_forms"][1] = "這兩日"
        second = json.loads(json.dumps(valid))
        second["keywords"][0]["doctor_evidence"] = "not in appointment"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.record_appointment.json").write_text(
                json.dumps({"notes": appointment}), encoding="utf-8"
            )
            (root / "sample.wav").write_bytes(b"RIFF-not-real-audio")
            config = root / "generated_keywords.json"
            with mock.patch.object(
                ev,
                "gemini_generate_keywords",
                side_effect=[first, second, valid],
            ) as generate:
                ev.generate_keyword_config(
                    root, config, "secret", "gemini-test"
                )
            third_feedback = generate.call_args_list[2].args[4]
        self.assertIn("non-Hong-Kong-Cantonese", third_feedback)
        self.assertIn("not an exact appointment excerpt", third_feedback)

    def test_keyword_csv_is_loaded_without_duplicate_canonical_form(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.record_appointment.json").write_text(
                json.dumps([{"diagnosis": "URTI", "History": "No fever"}]),
                encoding="utf-8",
            )
            (root / "sample.wav").write_bytes(b"RIFF-not-real-audio")
            config = root / "keyword_list.csv"
            config.write_text(
                "consultation_id,diagnosis,keyword,category,accepted_forms,"
                "doctor_evidence\n"
                "sample,URTI,發燒,症狀,發燒 | fever,No fever\n",
                encoding="utf-8-sig",
            )
            consultations = ev.load_consultations(root, config)
        keyword = consultations[0].keywords[0]
        self.assertEqual(keyword.text, "發燒")
        self.assertEqual(keyword.accepted_forms, ("fever",))

    def test_keyword_csv_requires_expected_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "keyword_list.csv"
            config.write_text("keyword\n發燒\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                ev.load_keyword_csv(config)

    def test_asr_model_selection(self):
        models = ev.parse_asr_models(
            "qwen3-asr-flash,qwen-audio-3.0-asr-flash"
        )
        self.assertEqual([model.backend for model in models], [
            "dashscope", "dashscope-audio3"
        ])
        with self.assertRaises(ValueError):
            ev.parse_asr_models("not-a-model")

    def test_dashscope_payloads_follow_playground_contracts(self):
        implementation = sys.modules[ev.dashscope_transcribe.__module__]
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "sample.wav"
            audio.write_bytes(b"audio")
            with mock.patch.object(implementation, "_post_json") as post:
                post.return_value = {
                    "choices": [{"message": {"content": "flash transcript"}}]
                }
                text = ev.dashscope_transcribe(
                    "secret", ev.ASR_MODELS["qwen3-asr-flash"], audio
                )
                payload = post.call_args.args[1]
            self.assertEqual(text, "flash transcript")
            self.assertTrue(
                payload["messages"][0]["content"][0]["input_audio"]["data"]
                .startswith("data:audio/wav;base64,")
            )

            with mock.patch.object(implementation, "_post_json") as post:
                post.return_value = {"output": {"text": "audio3 transcript"}}
                text = ev.dashscope_transcribe(
                    "secret", ev.ASR_MODELS["qwen-audio-3.0-asr-flash"], audio
                )
                payload = post.call_args.args[1]
            self.assertEqual(text, "audio3 transcript")
            self.assertEqual(payload["parameters"]["format"], "wav")


if __name__ == "__main__":
    unittest.main()
