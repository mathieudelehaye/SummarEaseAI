output "frontend_url" {
  value = "https://${azurerm_container_app.frontend.latest_revision_fqdn}"
}

output "backend_url" {
  value = "https://${var.huggingface_org}-${var.project_name}-backend.hf.space"
}

output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "container_registry_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "container_app_environment_id" {
  description = "The ID of the Container App Environment"
  value       = data.azurerm_container_app_environment.main.id
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