# SummarEaseAI - Azure Container Apps Deployment
# Terraform configuration for microservices with scale-to-zero capability

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
  }
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
  }
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.project_name}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
  }
}

# Azure Key Vault
resource "azurerm_key_vault" "main" {
  name                        = "${var.project_name}-kv"
  location                    = azurerm_resource_group.main.location
  resource_group_name         = azurerm_resource_group.main.name
  enabled_for_disk_encryption = true
  tenant_id                   = var.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false
  sku_name                   = "standard"

  access_policy {
    tenant_id = var.tenant_id
    object_id = var.client_id

    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete"
    ]
  }

  # Add access policy for the current user
  access_policy {
    tenant_id = var.tenant_id
    object_id = var.user_object_id

    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete"
    ]
  }

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
  }
}

# Store secrets in Key Vault
resource "azurerm_key_vault_secret" "openai_api_key" {
  name         = "openai-api-key"
  value        = var.openai_api_key
  key_vault_id = azurerm_key_vault.main.id
}

resource "azurerm_key_vault_secret" "backend_url" {
  name         = "backend-url"
  value        = "https://${azurerm_container_app.backend.latest_revision_fqdn}"
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_container_app.backend]
}

# Use existing Container App Environment instead of creating a new one
data "azurerm_container_app_environment" "main" {
  name                = var.existing_container_env_name
  resource_group_name = var.existing_container_env_rg
}

# Backend Container App
resource "azurerm_container_app" "backend" {
  name                         = "${var.project_name}-backend"
  container_app_environment_id = data.azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "backend"
      image  = "${azurerm_container_registry.main.login_server}/summarease-backend:latest"
      cpu    = 0.5
      memory = "2Gi"

      env {
        name  = "FLASK_ENV"
        value = "production"
      }

      env {
        name  = "AZURE_DEPLOYMENT"
        value = "true"
      }

      env {
        name        = "OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }

      liveness_probe {
        transport = "HTTP"
        port      = 5000
        path      = "/health"
      }

      readiness_probe {
        transport = "HTTP"
        port      = 5000
        path      = "/health"
      }
    }

    min_replicas = 0
    max_replicas = 5
  }

  ingress {
    external_enabled = true
    target_port      = 5000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username            = azurerm_container_registry.main.admin_username
    password_secret_name = "registry-password"
  }

  secret {
    name  = "registry-password"
    value = azurerm_container_registry.main.admin_password
  }

  secret {
    name  = "openai-api-key"
    value = var.openai_api_key
  }

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
    Service     = "Backend"
  }
}

# Frontend Container App
resource "azurerm_container_app" "frontend" {
  name                         = "${var.project_name}-frontend"
  container_app_environment_id = data.azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "frontend"
      image  = "${azurerm_container_registry.main.login_server}/summarease-frontend:latest"
      cpu    = 0.25
      memory = "0.5Gi"

      env {
        name  = "STREAMLIT_SERVER_PORT"
        value = "8501"
      }

      env {
        name  = "STREAMLIT_SERVER_ADDRESS"
        value = "0.0.0.0"
      }

      env {
        name  = "STREAMLIT_SERVER_HEADLESS"
        value = "true"
      }

      env {
        name        = "BACKEND_URL"
        secret_name = "backend-url"
      }

      liveness_probe {
        transport = "HTTP"
        port      = 8501
        path      = "/_stcore/health"
      }

      readiness_probe {
        transport = "HTTP"
        port      = 8501
        path      = "/_stcore/health"
      }
    }

    min_replicas = 0
    max_replicas = 3
  }

  ingress {
    external_enabled = true
    target_port      = 8501
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username            = azurerm_container_registry.main.admin_username
    password_secret_name = "registry-password"
  }

  secret {
    name  = "registry-password"
    value = azurerm_container_registry.main.admin_password
  }

  secret {
    name  = "backend-url"
    value = "https://${azurerm_container_app.backend.latest_revision_fqdn}"
  }

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
    Service     = "Frontend"
  }
} 