# Manual Workflow Setup Required

Due to GitHub App permission limitations, the following actions must be performed manually by repository maintainers:

## 1. Create Workflow Files

Copy the example workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow files
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
```

## 2. Configure Repository Secrets

The following secrets need to be configured in GitHub repository settings:

### Required Secrets

**PyPI Publishing:**
- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `TEST_PYPI_API_TOKEN`: Test PyPI API token for pre-release testing

**Docker Publishing:**
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password or access token

**Security Scanning:**
- `SNYK_TOKEN`: Snyk API token for container vulnerability scanning
- `GITLEAKS_LICENSE`: GitLeaks license key (if using pro version)

**Notifications:**
- `SLACK_WEBHOOK`: Slack webhook URL for CI notifications
- `SECURITY_SLACK_WEBHOOK`: Slack webhook for security alerts
- `DISCORD_WEBHOOK`: Discord webhook for release notifications (optional)

**Code Coverage:**
- `CODECOV_TOKEN`: Codecov token for coverage reporting

### Optional Secrets

**External Services:**
- `WANDB_API_KEY`: Weights & Biases API key for experiment tracking
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `DATADOG_API_KEY`: Datadog API key for metrics

## 3. Configure Branch Protection Rules

Enable the following branch protection rules for the `main` branch:

1. **Require pull request reviews before merging**
   - Required approving reviews: 1
   - Dismiss stale reviews when new commits are pushed: ✓
   - Require review from code owners: ✓

2. **Require status checks to pass before merging**
   - Require branches to be up to date before merging: ✓
   - Required status checks:
     - `Code Quality`
     - `Tests (ubuntu-latest, 3.11)`
     - `Security Scan / SAST`
     - `Security Scan / CodeQL Analysis`
     - `Package Build`

3. **Require conversation resolution before merging**: ✓

4. **Require signed commits**: ✓ (recommended)

5. **Require linear history**: ✓ (optional)

6. **Include administrators**: ✓

## 4. Configure Repository Settings

### General Settings

- **Allow merge commits**: ❌
- **Allow squash merging**: ✅
- **Allow rebase merging**: ✅
- **Automatically delete head branches**: ✅

### Security & Analysis

Enable the following features:

- **Dependency graph**: ✅
- **Dependabot alerts**: ✅
- **Dependabot security updates**: ✅
- **Dependabot version updates**: ✅
- **Code scanning alerts**: ✅
- **Secret scanning alerts**: ✅
- **Secret scanning push protection**: ✅

### Code Security and Analysis

**Dependabot Configuration** (create `.github/dependabot.yml`):

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "your-team"
    assignees:
      - "your-maintainer"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "deps"
      include: "scope"
```

## 5. Create GitHub Environments

Create the following environments in repository settings:

### Production Environment

- **Environment name**: `production`
- **Deployment protection rules**:
  - Required reviewers: Add team or specific users
  - Wait timer: 5 minutes
- **Environment secrets**:
  - `PYPI_API_TOKEN`
  - Production database credentials (if applicable)

### Staging Environment

- **Environment name**: `staging`
- **Deployment protection rules**: None (optional)
- **Environment secrets**:
  - `TEST_PYPI_API_TOKEN`
  - Staging environment credentials

## 6. Configure Issue Templates

Create `.github/ISSUE_TEMPLATE/` directory with the following templates:

**Bug Report** (`.github/ISSUE_TEMPLATE/bug_report.yml`):

```yaml
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: version
    attributes:
      label: Version
      description: What version of the software are you running?
      placeholder: e.g., 1.0.0
    validations:
      required: true
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Describe the bug
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to reproduce
      description: How can we reproduce this issue?
      placeholder: |
        1. Step one
        2. Step two
        3. Step three
    validations:
      required: true
```

**Feature Request** (`.github/ISSUE_TEMPLATE/feature_request.yml`):

```yaml
name: Feature Request
description: Suggest an idea for this project
title: "[Feature]: "
labels: ["enhancement", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a feature!
  - type: textarea
    id: feature-description
    attributes:
      label: Feature Description
      description: Describe the feature you'd like to see
      placeholder: What would you like to see implemented?
    validations:
      required: true
  - type: textarea
    id: motivation
    attributes:
      label: Motivation
      description: Why is this feature important?
      placeholder: What problem does this solve?
    validations:
      required: true
```

## 7. Configure Pull Request Template

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description

Brief description of the changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Checked code coverage

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## 8. Set up CODEOWNERS

Create `.github/CODEOWNERS`:

```
# Global owners
* @your-team

# Python code
*.py @python-experts

# CI/CD workflows  
.github/workflows/ @devops-team

# Documentation
docs/ @docs-team
*.md @docs-team

# Security files
SECURITY.md @security-team
.github/workflows/security.yml @security-team

# Dependencies
pyproject.toml @maintainers
requirements*.txt @maintainers
```

## 9. Verify Workflow Setup

After setting up the workflows, verify they work correctly:

1. **Create a test PR** to trigger CI workflows
2. **Check workflow runs** in the Actions tab
3. **Verify status checks** appear on PRs
4. **Test security scanning** by checking the Security tab
5. **Validate notifications** are sent to configured channels

## 10. Monitor and Maintain

- **Weekly**: Review failed workflow runs
- **Monthly**: Update action versions and dependencies
- **Quarterly**: Review and update security configurations
- **As needed**: Adjust workflow triggers and configurations based on team needs

## Troubleshooting

### Common Issues

1. **Workflow not triggering**: Check branch name patterns and triggers
2. **Permission errors**: Verify repository permissions and secrets
3. **Action version issues**: Pin to specific action versions
4. **Resource limits**: Use larger runners for resource-intensive jobs

### Getting Help

- Check GitHub Actions documentation
- Review workflow run logs
- Contact the development team
- File an issue in the repository

## Security Considerations

- **Rotate secrets regularly** (quarterly recommended)
- **Use least-privilege principles** for tokens and permissions
- **Monitor security alerts** and act on them promptly
- **Keep actions updated** to latest versions
- **Review third-party actions** before use

This setup ensures a robust, secure, and automated development workflow for the Self-Evolving MoE-Router project.