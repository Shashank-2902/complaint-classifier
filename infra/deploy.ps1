<#
Deploys the classifier API to Azure Container Apps.

NOT TESTED YET. Nobody has run this script. Read it first and run it yourself.

What it does, in this order:
  1. checks that you are logged in to Azure and that the active subscription is the expected one;
  2. asks you to type DEPLOY (nothing changes before that);
  3. terraform apply for the resource group and the container registry only (the image needs a registry to go to);
  4. az acr login, docker build, docker push;
  5. terraform apply for everything else (log workspace, environment, identity, container app).

The image is only set when the container app is first created: the Terraform file ignores later changes of the image,
because the deploy workflow updates it outside Terraform. To update the image of an existing app, see the end of this script's output.

Needs: Azure CLI (az login done), Terraform, Docker, and a model in deploy/model (made by scripts/export_registered_model.py).
Costs are not stated here: check them in the Azure portal.

Usage:  pwsh ./infra/deploy.ps1 [-ImageTag v1] [-ExpectedSubscription "Azure for Students"]
#>
param(
    [string]$ImageTag = "",
    [string]$ExpectedSubscription = "Azure for Students"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$tf = Join-Path $PSScriptRoot "terraform"

foreach ($tool in @("az", "terraform", "docker")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not installed or not on the PATH." }
}

# Terraform reads the subscription from this variable. An empty value means that you are not logged in.
$env:ARM_SUBSCRIPTION_ID = (az account show --query id -o tsv)
if (-not $env:ARM_SUBSCRIPTION_ID) {
    throw "No active Azure subscription. Log in first with: az login"
}
$subscriptionName = (az account show --query name -o tsv)
if ($subscriptionName -notlike "*$ExpectedSubscription*") {
    throw ("The active subscription is '$subscriptionName', not one that contains '$ExpectedSubscription'. " +
        "List your subscriptions with: az account list -o table  and switch with: az account set --subscription ""<name or id>""")
}

if (-not (Test-Path (Join-Path $root "deploy/model/tfidf_lr.joblib"))) {
    throw "deploy/model/tfidf_lr.joblib is missing. Run scripts/export_registered_model.py first."
}
if ($ImageTag -eq "") {
    $ImageTag = (git -C $root rev-parse --short HEAD 2>$null)
    if (-not $ImageTag) { $ImageTag = Get-Date -Format "yyyyMMddHHmm" }
}

Write-Host "Azure subscription: $subscriptionName ($env:ARM_SUBSCRIPTION_ID)"
Write-Host "Image tag: $ImageTag"
Write-Host "This creates Azure resources in the subscription above (resource group, container registry, log workspace,"
Write-Host "container apps environment, identity and the container app) and pushes a Docker image."
$answer = Read-Host "Type DEPLOY to continue"
if ($answer -ne "DEPLOY") { Write-Host "Cancelled. Nothing was changed."; exit 1 }

Push-Location $tf
try {
    terraform init -input=false
    # 1. only the resource group and the registry, so that the image has somewhere to go
    terraform apply -target=azurerm_resource_group.main -target=azurerm_container_registry.main

    $registry = terraform output -raw registry_name
    $server = terraform output -raw registry_login_server
    $image = "$server/complaint-classifier:$ImageTag"

    # 2. log in, build and push the image
    az acr login --name $registry
    docker build -t $image $root
    docker push $image

    # 3. everything else. The image tag is used when the app is created.
    terraform apply -var "image_tag=$ImageTag"

    $group = terraform output -raw resource_group_name
    $app = terraform output -raw container_app_name
    Write-Host ""
    Write-Host "App URL: $(terraform output -raw app_url)"
    Write-Host "If the app already existed before this run, its image was NOT changed (Terraform ignores it). Update it with:"
    Write-Host "  az containerapp update --name $app --resource-group $group --image $image"
}
finally {
    Pop-Location
}
