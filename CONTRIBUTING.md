<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to NuRec

Thank you for your interest in NuRec. Community contributions are welcome
within the scope described below, but please read this document first to
understand how the project operates.

NuRec is an NVIDIA-owned project: NVIDIA decides what is merged and shipped.
Roadmap planning and release scheduling happen at NVIDIA, and this public
repository publishes the source code.

**This GitHub repository does not merge pull requests directly.** A pull
request opened here is reviewed on GitHub. To land it, an NVIDIA engineer
imports the change into NVIDIA's source repository, where it goes through
NVIDIA's standard review and testing. The change then appears in this GitHub
repository with the next source update, and the original pull request is
closed (referencing the commit that landed it) rather than merged with
GitHub's merge button. This is the same shape as how
[Go handles GitHub pull requests via GerritBot](https://go.dev/doc/contribute):
review can happen where the contribution was opened, but the change lands
through a separate system. Expect turnaround to follow NVIDIA's update
cadence rather than immediate merge-on-approval.

All project participants must follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contribution scope

Accepted without prior discussion:

- Bug fixes. You may open a pull request directly alongside a bug report;
  you do not need to wait for maintainer sign-off.
- Small, focused improvements to code, tests, and documentation.

Please open an issue and discuss first for anything more substantial: a new
feature, a change to public APIs or configs, or a non-trivial refactor.
Changes that conflict with NVIDIA's roadmap may be declined even if they are
well implemented.

Issues track bugs and proposals; they are not a backlog of assignments up for
grabs. If you want to pick up an issue you didn't file yourself, say so in the
issue and confirm with a maintainer before investing significant effort, since
a given issue may already be scheduled, out of scope, or superseded by work in
progress at NVIDIA.

## Reporting bugs and proposing changes

Search existing [GitHub Issues](../../issues) and
[Discussions](../../discussions) for prior reports or related context before
opening a new one.

- Report bugs with the **Bug report** form (`bug_report`) when opening a new
  issue. Include reproduction steps, expected and actual behavior, the
  smallest practical reproducer, and relevant environment details.
- Report incorrect or needed documentation with the **Documentation request**
  form (`documentation_request`).
- Propose changes with the **Feature request** form (`feature_request`).
  Describe the problem it solves or the capability it adds, the impact on
  your use of NuRec (what you are building and what is blocked without it),
  the proposed approach, and any alternatives you considered.
- Prior discussion is required for new features, changes to public APIs or
  configs, and non-trivial refactors.
- For source questions, use the **Question / General Discussion** issue form
  (`question`).
- Usage questions and NuRec container help:
  [NVIDIA Developer Forum — NuRec](https://forums.developer.nvidia.com/c/omniverse/platform/nurec/752).
- Issue triage and priority: maintainers triage new issues and prioritize them
  against NVIDIA's roadmap. Priority is not guaranteed, and an issue may be
  labeled, redirected, or closed if it is out of scope.

## Development setup

- [Set up and build](README.md#set-up-and-build) for prerequisites and Bazel setup.
- [Tests and checks](#tests-and-checks) for formatting and testing.

### Tests and checks

The project is built and tested with Bazel. Before opening a pull request,
run:

```bash
bazel run //:format.check
bazel build ...
bazel test ...
```

To fix formatting violations automatically, use `bazel run //:format`. To run
a subset of the tests, scope the target, for example
`bazel test //nurec/datamodules/...`.

- Required hardware: an NVIDIA GPU and driver are needed to run the project
  and its GPU-dependent tests; see the
  [prerequisites](README.md#set-up-and-build).
- Checks available only to maintainers: after import, pull requests
  additionally go through NVIDIA's internal validation.

### Coding conventions

Follow the existing style and conventions of the file you are editing, and
keep changes focused. A pull request should address one bug fix, feature, or
improvement at a time.

### Adding or changing dependencies

Dependencies increase the maintenance, security, and licensing burden of the
project, so add them sparingly.

- **Avoid dependencies for minor needs.** Do not add a package for a small
  convenience or a few lines of functionality; implement it locally or use the
  standard library or an existing dependency instead.
- **Check the license.** The package, and everything it pulls in, must carry a
  license compatible with the project's Apache-2.0 license and with
  redistribution. Name the package and its license in the pull request. Be
  careful with copyleft licenses (GPL, AGPL, LGPL) and with packages that have
  no clear license; maintainers may decline them.
- **Open an issue first** for any new dependency; see
  [Contribution scope](#contribution-scope).

To change Python dependencies, edit the applicable `.in` file under
`deps/python` and regenerate the matching `.txt` lock file:

- `requirements_3_13_common.in`: architecture-independent requirements.
- `requirements_3_13_x86_64.in`: x86_64-specific requirements.
- `requirements_3_13_docs.in`: documentation-only requirements.

Then regenerate the lock file for the edited input, for example:

```bash
bazel run //deps/python:requirements_3_13_x86_64.update
```

Commit both the `.in` and the regenerated `.txt` files. A test fails when a
lock file is out of date with its `.in` input.

## Pull request process

When opening a pull request:

- Include a clear description of the change: what it does, why, and how it
  was tested. Reference any related issue.
- Keep pull requests focused on a single change; split unrelated changes into
  separate pull requests.
- Add test cases for new code paths and make sure the checks above pass.
- Be responsive during review. Reviewers may ask clarifying questions or
  request changes before a contribution is accepted.

Commit messages must follow
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

What to expect once a pull request is approved:

1. An NVIDIA engineer imports the change into NVIDIA's source repository,
   where it goes through NVIDIA's standard review and testing.
2. The change appears in this GitHub repository with the next source update.
3. The original GitHub pull request is closed, referencing the commit that
   landed it, rather than merged directly on GitHub.

## Legal contribution requirement

### Developer Certificate of Origin

Sign off every commit with `git commit -s` (or `--signoff`). By signing off,
you certify that your contribution complies with the
[Developer Certificate of Origin](https://developercertificate.org/). NuRec
uses the DCO rather than a Contributor License Agreement (CLA). The full text
of the certificate is:

```text
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Review process

Pull requests are reviewed by the NuRec maintainers. Maintainers aim to
respond to a new pull request within 2 weeks. If you have not received a
response, comment on the pull request to request a status update. A pull
request may be closed if it has been inactive for 60 days without response to
maintainer feedback.

## AI-assisted contributions

- **You are responsible:** Review and understand all AI-generated code before
  submitting it.
- **Verify correctness:** AI-generated code may contain bugs or security
  issues. Run the applicable tests and review the diff carefully.
- **No AI-generated sensitive content:** Do not use AI to generate security
  policies, legal, or license text.

## Security issues

Do not report security vulnerabilities through public issues or pull requests.
Follow the process in [SECURITY.md](SECURITY.md).
