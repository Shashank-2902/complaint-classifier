# Week 3: tracking a model, putting it in a container and describing its hosting as code

Nothing in this note was deployed. The Docker image was built and run on localhost, and the Terraform files pass `terraform validate`. The Azure apply and the deploy workflow were not run.

## 1. Concept

### Why track a model at all

A model file on a laptop is just a file. Six months later nobody knows which data it saw, which settings it used or how good it was. **Experiment tracking** writes this down for every training run:
the parameters, the scores, the data version and the code version. MLflow does this in a local folder. A **model registry** adds names and version numbers on top: "complaint-classifier-tfidf, version 1".

In this project the data version is the **fingerprint** of every split (a short code computed from the list of complaint ids, stored in `results/data_report.json`). The code version is the git commit.

### Validation numbers and test numbers stay apart

The tracking script scores only the validation split. It fits the model on the training part and scores it on the validation part (accuracy 0.7303, macro-F1 0.6928 in the run of 2026-10-26).
The test numbers that were published earlier are copied from `results/results.md` into tags named `published_test_*`, with the tag `source`. The script never computes a test number, so tracking cannot touch the test set.

The registered model is a second fit, on training plus validation, because that is how the API model was made. So the registered model has no honest validation score of its own. The model card says this.

### Container, image and health check

A **Docker image** holds the code, the Python packages and the model, so it runs the same everywhere. The image here uses Python 3.11 slim, runs as a user that is not root and contains only the packages the API needs
(`requirements-api.txt`). A **health check** is a command that Docker runs now and then. If `/health` does not answer 200 (for example because the model file is missing), the container is marked unhealthy.

### Infrastructure as code

**Terraform** describes the cloud resources in text files. `terraform validate` checks the files without an Azure account. `terraform plan` would show what would change and `terraform apply` would create it. Only the first one
is safe to run without knowing the account, and neither was run here.

The resources: a resource group, a container registry, a log workspace with a small daily quota, a Container Apps environment and a container app. The app pulls its image with a **managed identity** that has the `AcrPull` role on the registry.
That means there is no password for the registry anywhere.

### Scale to zero and its price

The app is allowed 0 to 1 replicas. With 0 replicas nothing runs while nobody calls the API, which is cheap, but the first call after a quiet period waits for a container to start (a **cold start**). The endpoint is public and has no login,
so the text length is limited and there is at most one replica. What it costs must be checked in the Azure portal.

### OIDC instead of a stored password

A GitHub workflow can log in to Azure with **OpenID Connect**. Azure trusts tokens that GitHub issues for one named repository and environment (a **federated credential**). The workflow stores only three identifiers as secrets
(client id, tenant id, subscription id), not a password. The steps to set this up are in `infra/README.md` and are marked not tested yet.

## 2. Code walkthrough

**`scripts/track_tfidf.py`**: `track()` fits the model with the same settings and seed as `src/baselines`, scores the validation part, fits the final model, logs parameters, metrics, the 12 data fingerprints,
the git commit and the published test tags to a local MLflow database in `mlruns/` (not committed), and registers the model. `published_test_numbers()` reads the TF-IDF row from `results/results.md`.

**`scripts/export_registered_model.py`**: `export()` loads a registered version, builds the same bundle structure as `models/tfidf_lr.joblib` and, with `--compare`, checks that the predictions on the first 1,000 validation rows are the same as the old model
before it writes anything. It writes `deploy/model/tfidf_lr.joblib` (12,733,976 bytes, compressed), `model_info.json` and `MODEL_CARD.md`.

**`api/main.py`**: the model folder comes from `MODEL_DIR` (default `deploy/model`). `/health` returns the status and the model version, `/classify` also returns the version, and a text of more than 5,000 characters gets a 422 with a clear message.
The model is loaded when the server starts.

**`Dockerfile`, `.dockerignore`, `requirements-api.txt`**: the image. The `.dockerignore` keeps data, results, models, tests, notebooks and secrets out of the build.

**`infra/terraform/`**: `versions.tf` (provider pinned to azurerm 4.81.0), `variables.tf`, `main.tf` (the resources), `outputs.tf` (the app URL). **`infra/deploy.ps1`** and **`infra/teardown.ps1`** ask for a typed confirmation first.
`deploy.ps1` applies the resource group and the registry first, because the image needs a registry to go to, then pushes the image, then applies the rest.
Both scripts first set `ARM_SUBSCRIPTION_ID` from the Azure CLI, stop if nobody is logged in and stop unless the subscription name contains "Azure for Students". If the resource group survives `terraform destroy`
(for example when the local state was lost), `teardown.ps1` asks for a second confirmation and deletes the group with `az group delete`.
In `main.tf` the container app ignores later changes of its image (`ignore_changes`), because the deploy workflow updates the image with `az containerapp update`, outside Terraform.

**CI**: `.github/workflows/ci.yml` has two new jobs, `docker-build` (build, start, check `/health`) and `terraform-validate` (format check, init without a backend, validate; no credentials).
`.github/workflows/deploy.yml` runs only on tags `v*` and by hand.

**Tests**: the API tests (health, version, model folder, text limit) need no Docker and no `data/` folder. The MLflow script tests run on the small fixture data and are skipped when MLflow is not installed.

## 3. Interview questions

1. **Why did you track the model with MLflow when it is only a small classifier?**
   So that a model file always comes with the data it saw (fingerprints), the settings, the code version and its scores. The registry gives it a name and a version number, and the exported model card says which version is in the image.

2. **How do you make sure the tracking does not leak the test set?**
   The script scores only the validation split. The published test numbers are copied from `results/results.md` as tags with their source, and no test number is computed. The test-set guard is not called at all.

3. **Why are there two fits?**
   The validation score needs a model that has not seen the validation rows, so it is fitted on the training part only. The registered model is fitted on training plus validation, like the model the API served before. It therefore has no honest validation score,
   and the model card says so.

4. **How did you check that the exported model is the same as the old one?**
   The export script compares the predicted probabilities of both models on the first 1,000 validation rows and stops without writing anything if they differ. In the real run it printed that the predictions were the same.

5. **Why is the model inside the Docker image?**
   It is 12.7 MB, so it is simple to ship it with the code. Then the container has nothing to download at start, and the image version and the model version cannot drift apart. A bigger model would be better kept outside the image.

6. **Why a non-root user and a health check?**
   If someone breaks into the app, a non-root user has fewer rights. The health check makes Docker (and Azure) notice when the app is up but cannot answer, for example when the model is missing.

7. **Why a managed identity with AcrPull instead of a registry password?**
   There is no secret to store, leak or rotate. Azure itself vouches for the app, and the role gives it the right to pull images and nothing else.

8. **Why does `deploy.ps1` apply Terraform twice?**
   The container app needs an image in the registry, and the registry must exist before an image can be pushed. So the first apply is limited to the resource group and the registry, then the image is pushed, then everything else is applied.
   The image is only set when the app is first created, because the Terraform file ignores later image changes (the deploy workflow updates the image outside Terraform). To change the image later, use `az containerapp update --image ...`.

9. **What are the risks of this deployment?**
   Scale to zero means a cold start. The endpoint is public without a login, so it has a text size limit and one replica. The student credit expires. The deploy workflow, the Terraform apply and the OIDC steps were never run, so they may need fixes.

10. **What is the difference between what you tested and what you did not?**
    Tested: the API tests, the MLflow scripts on fixture data and on the real data, the Docker image on localhost, `terraform fmt`, `init` and `validate`, and the syntax of the workflow files and the PowerShell scripts.
    Not tested: `terraform plan` and `apply`, the Azure deploy, the deploy workflow, the OIDC credentials and the cost.

## 4. Glossary

- **Experiment tracking**: saving parameters, scores, data and code versions of every training run.
- **Model registry**: a list of named, versioned models.
- **Fingerprint**: a short code computed from a list of ids. The same fingerprint means the same split.
- **Model card**: a short text about a model: version, data, metrics, limits.
- **Docker image / container**: a packaged program with everything it needs / a running copy of an image.
- **Health check**: a command that tells whether a container works.
- **Terraform**: a tool that creates cloud resources from text files.
- **Provider**: the plugin that lets Terraform talk to one cloud (here azurerm).
- **Managed identity**: an identity that Azure gives to an app, so it needs no password.
- **AcrPull**: the Azure role that allows pulling images from a registry.
- **Scale to zero / cold start**: no replica while idle / the wait for the first request afterwards.
- **OIDC (federated credential)**: a login where Azure trusts a token from GitHub, so no password is stored.
