output "app_url" {
  description = "Public URL of the API. Open /docs for the Swagger page."
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "registry_name" {
  description = "Name of the container registry (used by infra/deploy.ps1)."
  value       = azurerm_container_registry.main.name
}

output "registry_login_server" {
  description = "Login server of the container registry."
  value       = azurerm_container_registry.main.login_server
}

output "resource_group_name" {
  description = "Resource group that holds everything (used by infra/teardown.ps1)."
  value       = azurerm_resource_group.main.name
}

output "container_app_name" {
  description = "Name of the container app (used to update its image with az containerapp update)."
  value       = azurerm_container_app.api.name
}
