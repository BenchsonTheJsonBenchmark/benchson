import json
import os
import re


class BaseDataTask:

    def __init__(self, system_prompt_prefix=""):
        self.system_prompt_prefix = system_prompt_prefix

    def _prepend_prefix(self, content):
        if self.system_prompt_prefix:
            return self.system_prompt_prefix + "\n" + content
        return content

    def load_instances(self, source=None, filter_kw=None, files=None, max_schema_kb=10):
        """Returns a list of schema file paths after applying source/filter/files config."""
        if files:
            result = []
            for p in files:
                path = str(p)
                result.append(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
            return result

        folder = os.path.join(os.getcwd(), "data", str(source), "test")
        if not os.path.isdir(folder):
            raise ValueError(f"Dataset folder not found: {folder}")

        paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".json")
        ]

        if filter_kw:
            paths = [p for p in paths if self._matches_filter(p, filter_kw)]

        if max_schema_kb is not None:
            limit = max_schema_kb * 1024
            paths = [p for p in paths if os.path.getsize(p) <= limit]

        if not paths:
            raise ValueError(
                f"No instances found in '{folder}' matching filter '{filter_kw}'"
            )

        return paths

    def _matches_filter(self, path, keyword):
        keyword = keyword.lower()
        if keyword in os.path.basename(path).lower():
            return True
        try:
            with open(path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            title = schema.get("title", "")
            return keyword in title.lower()
        except Exception:
            return False

    def generate_valid_json(self, schema_path, llm_provider, max_retries):
        """Runs the LLM with a self-fix loop to produce valid JSON for the schema.

        Returns (schema_dict, valid_json_str, initial_messages) or None on failure.
        """
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        initial_messages = self._build_schema_creation_prompt(schema)
        result = self._run_fix_loop(schema, initial_messages, llm_provider, max_retries)
        if result is None:
            return None
        valid_json_str, messages = result
        return schema, valid_json_str, messages

    def _run_fix_loop(self, schema, initial_messages, llm_provider, max_retries):
        """Generic self-fix loop given pre-built initial messages.

        Returns (valid_json_str, initial_messages) or None on failure.
        """
        messages = list(initial_messages)

        for attempt in range(max_retries + 1):
            raw, _ = llm_provider.generate(messages)
            cleaned = self._clean_output(raw)

            error = self._validate_json(cleaned, schema)
            if error is None:
                return cleaned, initial_messages

            if attempt < max_retries:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        f"The JSON is invalid: {error[:300]}. "
                        "Fix it and output only the corrected JSON."
                    )},
                ]

        return None

    def format_example(self, messages, output_str, fmt):
        if fmt == "alpaca":
            user_content = next(m["content"] for m in messages if m["role"] == "user")
            return {"instruction": user_content, "input": "", "output": output_str}

        # chat format (default)
        training_messages = [m for m in messages if m["role"] != "assistant" or m["content"]]
        # replace the trailing empty assistant turn with the real output
        if messages and messages[-1]["role"] == "assistant" and messages[-1]["content"] == "":
            training_messages = messages[:-1] + [{"role": "assistant", "content": output_str}]
        else:
            training_messages = messages + [{"role": "assistant", "content": output_str}]
        return {"messages": training_messages}

    def generate(self, schema_path, llm_provider, max_retries):
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers

    # Appended to the creation prompt in complex Create mode only. A first wording to be
    # reviewed on real model output; the standard prompt is never touched.
    COMPLEXITY_HINT = (
        "\n\nMake the object rich and realistic: fill in the optional properties the schema "
        "defines (not only the required ones), complete nested objects, give every array at "
        "least two items - and make the items of an array of objects differ from each other - "
        "and use specific, varied values instead of placeholders. It must still conform "
        "exactly to the schema."
    )

    def _build_schema_creation_prompt(self, schema, complexity_hint=False):
        prompt = self._build_plain_schema_creation_prompt(schema)
        if complexity_hint:
            prompt[1] = {**prompt[1], "content": prompt[1]["content"] + self.COMPLEXITY_HINT}
        return prompt

    def _build_plain_schema_creation_prompt(self, schema):
        return [
            {
                "role": "system",
                "content": self._prepend_prefix("You are a helpful assistant that generates JSON data based on a given schema. Output only the json with no other text or explanations. Always populate fields with realistic values — never return an empty object or array when the schema allows properties."),
            },
            {
                "role": "user",
                "content": f"Generate a JSON object that conforms to the following schema: {json.dumps(schema, indent=2)}",
            },
            {"role": "assistant", "content": ""},
        ]

    def _build_described_schema_creation_prompt(self, schema, description):
        return [
            {
                "role": "system",
                "content": self._prepend_prefix("You are a helpful assistant that generates JSON data. Output only the JSON with no other text or explanations. Always populate fields with realistic values — never return an empty object or array when the schema allows properties."),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a JSON object representing: {description}\n\n"
                    f"The JSON must conform to the following schema:\n{json.dumps(schema, indent=2)}"
                ),
            },
            {"role": "assistant", "content": ""},
        ]

    def _build_source_doc_prompt(self, schema, reference_json):
        return [
            {
                "role": "user",
                "content": (
                    "You are an assistant that writes natural language documents containing "
                    "structured information.\n\n"
                    "You will receive two inputs:\n"
                    f"1. A JSON Schema describing the structure of some data:\n{json.dumps(schema, indent=2)}\n\n"
                    "2. A valid JSON object conforming to that schema, containing the actual values:\n"
                    f"{json.dumps(reference_json, indent=2)}\n\n"
                    "Your task is to write a short natural language document (1-3 paragraphs) that "
                    "contains all the information present in the JSON object. The document should read "
                    "naturally — like a real email, report, config description, or data entry — not like "
                    "a mechanical list of field values.\n\n"
                    "Requirements:\n"
                    "- Every field value in the reference JSON must be recoverable from the document.\n"
                    "- Do not invent values that are not in the reference JSON.\n"
                    "- Do not mention the schema or JSON format in the document.\n"
                    "- The document should be realistic and domain-appropriate given the schema's subject matter.\n\n"
                    "Output only the document text, no explanations or formatting."
                ),
            },
        ]

    def _build_extraction_prompt(self, schema, source_doc):
        return [
            {
                "role": "system",
                "content": self._prepend_prefix(
                    "You are a helpful assistant that extracts information from text into JSON data "
                    "matching a given schema. Output only the JSON with no other text or explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract the information from the following text and return it as a JSON object "
                    "that conforms to the schema.\n\n"
                    f"Text:\n{source_doc}\n\n"
                    f"Schema:\n{json.dumps(schema, indent=2)}"
                ),
            },
            {"role": "assistant", "content": ""},
        ]

    def _build_error_correction_prompt(self, schema, erroneous_json):
        return [
            {
                "role": "system",
                "content": self._prepend_prefix(
                    "You are a helpful assistant tasked with fixing JSON objects to conform precisely "
                    "to the provided JSON schema. Return ONLY the corrected JSON object, "
                    "with no additional text or explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Please correct the following JSON so that it fully matches the given schema:\n\n"
                    f"JSON:\n```json\n{json.dumps(erroneous_json, indent=2)}\n```\n\n"
                    f"Schema:\n```json\n{json.dumps(schema, indent=2)}\n```"
                ),
            },
            {"role": "assistant", "content": ""},
        ]

    def _clean_output(self, text):
        return re.sub(r"```json\s*|\s*```", "", text).strip()

    def _validate_json(self, text, schema):
        """Returns None if valid, error string if invalid."""
        import jsonschema
        try:
            obj = json.loads(text)
            jsonschema.validate(instance=obj, schema=schema)
            return None
        except json.JSONDecodeError as e:
            return f"JSON parse error: {e}"
        except jsonschema.ValidationError as e:
            return str(e.message)
        except Exception as e:
            # Unresolvable $ref, malformed schema, etc. — treat as invalid so the
            # fix loop keeps trying (and unusable schemas are skipped, not fatal).
            return f"schema error: {e}"
