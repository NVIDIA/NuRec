# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

import yaml

from jinja2 import Environment, FileSystemLoader, select_autoescape


def main():
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate entrypoint script from a template.")
    parser.add_argument(
        "--config", required=True, nargs="+", help="Path(s) to the tools YAML config file(s), merged together."
    )
    parser.add_argument("--template", required=True, help="Path to the entrypoint template file.")
    parser.add_argument("--output", required=True, help="Path to the output entrypoint script.")
    parser.add_argument("--enabled-tools", nargs="*", default=[], help="List of enabled conditional tools by name.")
    args = parser.parse_args()

    all_tools = []
    for config_path in args.config:
        logging.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        all_tools.extend(config.get("tools", []))
    enabled_tools_set = set(args.enabled_tools)
    tool_names = [tool["name"] for tool in all_tools]
    logging.info(f"Tools present = {', '.join(tool_names) if tool_names else 'none'}")
    logging.info(f"Tools enabled = {', '.join(enabled_tools_set) if enabled_tools_set else 'none'}")

    final_tools = []
    for tool in all_tools:
        if tool["name"] in enabled_tools_set:
            final_tools.append(tool)

    # The template path from bazel is relative to the execroot, so we can use it to find the template dir
    template_dir = os.path.dirname(args.template)
    template_name = os.path.basename(args.template)
    env = Environment(
        loader=FileSystemLoader(template_dir if template_dir else "."),
        autoescape=select_autoescape(),
    )
    template = env.get_template(template_name)

    output_content = template.render(tools=final_tools)

    tool_names = [tool["name"] for tool in final_tools]
    logging.info(f"Writing tools: {', '.join(tool_names) if tool_names else 'none'} to entrypoint to {args.output}")
    with open(args.output, "w") as f:
        f.write(output_content)

    logging.info("Entrypoint generation completed successfully")


if __name__ == "__main__":
    main()
