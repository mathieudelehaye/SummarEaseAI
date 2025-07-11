output "resource_group_name" {
  description = "The name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "container_registry_login_server" {
  description = "The login server URL for the container registry"
  value       = azurerm_container_registry.main.login_server
}

output "container_app_environment_id" {
  description = "The ID of the Container App Environment"
  value       = data.azurerm_container_app_environment.main.id
}

output "backend_url" {
  description = "The URL of the backend service"
  value       = "https://${azurerm_container_app.backend.latest_revision_fqdn}"
}

output "frontend_url" {
  description = "The URL of the frontend service"
  value       = "https://${azurerm_container_app.frontend.latest_revision_fqdn}"
}

output "deployment_commands" {
  description = "Commands to build and push the Docker images"
  value = {
    backend  = "docker build -t ${azurerm_container_registry.main.login_server}/summarease-backend:latest -f backend/Dockerfile . && docker push ${azurerm_container_registry.main.login_server}/summarease-backend:latest"
    frontend = "docker build -t ${azurerm_container_registry.main.login_server}/summarease-frontend:latest -f frontend/Dockerfile . && docker push ${azurerm_container_registry.main.login_server}/summarease-frontend:latest"
  }
}

output "log_analytics_workspace_id" {
  description = "The ID of the Log Analytics workspace"
  value       = azurerm_log_analytics_workspace.main.id
} 