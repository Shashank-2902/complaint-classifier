terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "= 4.81.0"
    }
  }
}

# The subscription comes from the environment (ARM_SUBSCRIPTION_ID) or from the Azure CLI login.
# No credentials are written in any file.
provider "azurerm" {
  features {}
}
