# Deploying the API to Azure Container Apps

Nothing in this folder has been run against Azure. `terraform fmt -check`, `terraform init -backend=false` and `terraform validate` pass (they need no Azure account).
`terraform plan`, `terraform apply`, `infra/deploy.ps1`, `infra/teardown.ps1` and `.github/workflows/deploy.yml` were **not run**.
Costs must be checked in the Azure portal. This README states no price.

## What Terraform creates (`infra/terraform/`)

| Resource | Notes |
| --- | --- |
| Resource group | `rg-<name_prefix>`, region `germanywestcentral` by default |
| Container registry | SKU Basic, admin user off |
| Log Analytics workspace | daily quota 0.1 GB, retention 30 days (the shortest this SKU allows) |
| Container Apps environment | logs go to the workspace |
| User-assigned managed identity | role AcrPull on the registry, so the app pulls the image without a password |
| Container app | external HTTPS ingress, port 8000, 0.25 vCPU, 0.5 Gi, 0 to 1 replicas, liveness probe on `/health` |

All resources carry the tags `project` and `managed`. The output `app_url` is the public address. There are no secrets in any file.
The provider is pinned to `azurerm` 4.81.0. The registry name is built from the prefix and a short hash of the subscription id, because registry names must be unique in Azure.

## Deploy and tear down

Prerequisites: the Azure CLI with `az login` done and the right subscription selected, Terraform, Docker, and a model in `deploy/model`
(made by `scripts/export_registered_model.py`).

```powershell
pwsh ./infra/deploy.ps1        # asks you to type DEPLOY, then runs the steps below
pwsh ./infra/teardown.ps1      # asks you to type DESTROY, then destroys and checks that the resource group is gone
```

Both scripts start by setting `ARM_SUBSCRIPTION_ID` from `az account show` and stop with a message if you are not logged in. They also stop unless the name of the active
subscription contains `Azure for Students` (change it with `-ExpectedSubscription "<part of the name>"`). To switch subscription: `az account list -o table`, then
`az account set --subscription "<name or id>"`. If the resource group still exists after `terraform destroy` (for example because the local Terraform state was lost), `teardown.ps1` asks for a
second typed confirmation, `DELETE-GROUP`, and then runs `az group delete --name <group> --yes`.

`deploy.ps1` does this, in order: `terraform apply` for the resource group and the registry only; `az acr login`, `docker build`, `docker push`; then the full `terraform apply`
with the image tag. The registry has to exist before the image can be pushed, which is why the first apply is limited with `-target`.

If Terraform complains that a resource provider is not registered, the subscription may need `Microsoft.App`, `Microsoft.OperationalInsights` and
`Microsoft.ContainerRegistry` registered (in the portal under Subscriptions, Resource providers). This was not needed or checked here.

### How the image is updated

The container app in `main.tf` has `lifecycle { ignore_changes = [template[0].container[0].image] }`. The image is therefore only set when the app is **first created**, from the variable `image_tag`.
Later changes of that variable are ignored, because the deploy workflow updates the image outside Terraform with `az containerapp update`, and Terraform must not set it back.
That is also why `deploy.ps1` does not change the image of an app that already exists. To update it by hand (the CLI extension is installed with `az extension add --name containerapp --upgrade --yes`):

```powershell
az containerapp update --name <container_app_name> --resource-group <resource_group_name> --image <registry_login_server>/complaint-classifier:<tag>
```

The three names are Terraform outputs (`terraform output`), and `deploy.ps1` prints the command at the end. Not tested yet.

## OIDC credentials for the deploy workflow (not tested yet)

The workflow `.github/workflows/deploy.yml` logs in to Azure without a stored password, using OpenID Connect. The steps below are written from the documentation of the Azure CLI and
GitHub. **Not tested yet.** Run them yourself and check each result.

1. Create an app registration and a service principal (note the `appId` it prints):

   ```powershell
   az ad app create --display-name complaint-classifier-deploy
   az ad sp create --id <appId>
   ```

2. In the GitHub repository, create an environment called `azure` (Settings, Environments). The workflow job uses it, and the credential below is tied to it.

3. Create the federated credential for that environment:

   ```powershell
   az ad app federated-credential create --id <appId> --parameters '{
     "name": "github-azure-environment",
     "issuer": "https://token.actions.githubusercontent.com",
     "subject": "repo:Shashank-2902/complaint-classifier:environment:azure",
     "audiences": ["api://AzureADTokenExchange"]
   }'
   ```

4. Give the service principal the rights it needs, only on the resource group that Terraform creates (after the first deploy):

   ```powershell
   az role assignment create --assignee <appId> --role Contributor --scope /subscriptions/<subscriptionId>/resourceGroups/rg-complaintclf
   az role assignment create --assignee <appId> --role AcrPush --scope /subscriptions/<subscriptionId>/resourceGroups/rg-complaintclf
   ```

5. Add three secrets to the GitHub repository (Settings, Secrets and variables, Actions): `AZURE_CLIENT_ID` (the `appId`), `AZURE_TENANT_ID` and `AZURE_SUBSCRIPTION_ID`.

6. Start the workflow: push a tag such as `v1` or run it by hand under Actions, Deploy.

The workflow name prefix in `deploy.yml` (`rg-complaintclf`, `ca-complaintclf`) assumes the default `name_prefix`. Change both if you change the variable.
