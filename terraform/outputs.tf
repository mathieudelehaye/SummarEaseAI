output "resource_group_name" {
  description = "Name of the created resource group"
  value       = azurerm_resource_group.main.name
}

output "container_registry_login_server" {
  description = "Login server URL of the container registry"
  value       = azurerm_container_registry.main.login_server
}

output "backend_url" {
  description = "URL of the backend API"
  value       = "https://${azurerm_container_app.backend.latest_revision_fqdn}"
}

output "frontend_url" {
  description = "URL of the frontend application"
  value       = "https://${azurerm_container_app.frontend.latest_revision_fqdn}"
}

output "container_app_environment_id" {
  description = "ID of the Container Apps Environment"
  value       = azurerm_container_app_environment.main.id
}

output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics Workspace"
  value       = azurerm_log_analytics_workspace.main.id
}

output "deployment_commands" {
  description = "Commands to deploy Docker images"
  value = {
    backend = "docker build -f backend/Dockerfile -t ${azurerm_container_registry.main.login_server}/summarease-backend:latest . && docker push ${azurerm_container_registry.main.login_server}/summarease-backend:latest"
    frontend = "docker build -f frontend/Dockerfile -t ${azurerm_container_registry.main.login_server}/summarease-frontend:latest . && docker push ${azurerm_container_registry.main.login_server}/summarease-frontend:latest"
  }
} 