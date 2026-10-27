data "azurerm_client_config" "current" {}

locals {
  # Registry names must be unique in all of Azure. A short hash of the subscription id makes the name stable and unique.
  registry_name = "${var.name_prefix}${substr(md5(data.azurerm_client_config.current.subscription_id), 0, 8)}"
  image         = "${azurerm_container_registry.main.login_server}/complaint-classifier:${var.image_tag}"
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${var.name_prefix}"
  location = var.location
  tags     = var.tags
}

# The admin user is off: the app pulls images with a managed identity (see AcrPull below).
resource "azurerm_container_registry" "main" {
  name                = local.registry_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false
  tags                = var.tags
}

# Small daily quota and the shortest retention that this SKU allows, so logs cannot grow unnoticed.
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${var.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  daily_quota_gb      = 0.1
  tags                = var.tags
}

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${var.name_prefix}"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = var.tags
}

resource "azurerm_user_assigned_identity" "app" {
  name                = "id-${var.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = var.tags
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

resource "azurerm_container_app" "api" {
  name                         = "ca-${var.name_prefix}"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = var.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.app.id
  }

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "api"
      image  = local.image
      cpu    = 0.25
      memory = "0.5Gi"

      liveness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8000
      }
    }
  }

  # The role must exist before the app tries to pull the image.
  depends_on = [azurerm_role_assignment.acr_pull]

  # The deploy workflow updates the image outside Terraform (az containerapp update). Without this,
  # the next terraform apply would set the image back to the value of var.image_tag.
  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }
}
