<#
Removes everything that infra/deploy.ps1 created.

NOT TESTED YET. Nobody has run this script. Read it first and run it yourself.

What it does:
  1. checks that you are logged in to Azure and that the active subscription is the expected one;
  2. asks you to type DESTROY (nothing changes before that);
  3. terraform destroy;
  4. checks that the resource group is gone. If it is still there (for example because the local Terraform state was lost),
     it asks you to type DELETE-GROUP and then deletes the resource group with the Azure CLI.

Usage:  pwsh ./infra/teardown.ps1 [-ExpectedSubscription "Azure for Students"]
#>
param(
    [string]$ExpectedSubscription = "Azure for Students"
)

$ErrorActionPreference = "Stop"
$tf = Join-Path $PSScriptRoot "terraform"

foreach ($tool in @("az", "terraform")) {
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

Write-Host "Azure subscription: $subscriptionName ($env:ARM_SUBSCRIPTION_ID)"
Write-Host "This deletes every resource that Terraform created for this project, including the container registry and its images."
$answer = Read-Host "Type DESTROY to continue"
if ($answer -ne "DESTROY") { Write-Host "Cancelled. Nothing was changed."; exit 1 }

Push-Location $tf
try {
    # If the local state is lost this output is empty, so the default name of the resource group is used.
    $group = terraform output -raw resource_group_name 2>$null
    if (-not $group) { $group = "rg-complaintclf" }
    terraform destroy
}
finally {
    Pop-Location
}

if ((az group exists --name $group) -eq "false") {
    Write-Host "Checked: the resource group $group is gone."
    exit 0
}

Write-Host "The resource group $group still exists. The local Terraform state may be lost or incomplete."
Write-Host "Deleting the group with the Azure CLI removes everything inside it."
$second = Read-Host "Type DELETE-GROUP to delete the resource group $group"
if ($second -ne "DELETE-GROUP") {
    Write-Host "Not deleted. Check the resource group $group in the Azure portal."
    exit 1
}
az group delete --name $group --yes

if ((az group exists --name $group) -eq "false") {
    Write-Host "Checked: the resource group $group is gone."
}
else {
    Write-Host "WARNING: the resource group $group still exists. Check it in the Azure portal."
    exit 1
}
