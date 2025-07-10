# SummarEaseAI - Azure Container Apps Deployment
# Terraform configuration for microservices with scale-to-zero capability

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~>2.0"
    }
  }
}

provider "azurerm" {
  features {}
}

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

# Container Apps Environment
resource "azurerm_container_app_environment" "main" {
  name                       = "${var.project_name}-env"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
  }
}

# Backend Container App
resource "azurerm_container_app" "backend" {
  name                         = "${var.project_name}-backend"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "backend"
      image  = "${azurerm_container_registry.main.login_server}/summarease-backend:latest"
      cpu    = 1.0
      memory = "2Gi"

      env {
        name  = "FLASK_ENV"
        value = "production"
      }

      env {
        name  = "TF_CPP_MIN_LOG_LEVEL"
        value = "3"
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
        http_get {
          path = "/health"
          port = 5000
        }
        initial_delay_seconds = 30
        period_seconds        = 30
      }

      readiness_probe {
        http_get {
          path = "/health"
          port = 5000
        }
        initial_delay_seconds = 10
        period_seconds        = 5
      }
    }

    min_replicas = 0  # Scale to zero when not in use
    max_replicas = 5  # Scale up to 5 instances under load
  }

  ingress {
    external_enabled = true
    target_port      = 5000
    traffic_weight {
      percentage = 100
      latest_revision = true
    }
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
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
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "frontend"
      image  = "${azurerm_container_registry.main.login_server}/summarease-frontend:latest"
      cpu    = 0.5
      memory = "1Gi"

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
        name  = "API_BASE_URL"
        value = "https://${azurerm_container_app.backend.latest_revision_fqdn}"
      }

      liveness_probe {
        http_get {
          path = "/_stcore/health"
          port = 8501
        }
        initial_delay_seconds = 30
        period_seconds        = 30
      }

      readiness_probe {
        http_get {
          path = "/_stcore/health"
          port = 8501
        }
        initial_delay_seconds = 10
        period_seconds        = 5
      }
    }

    min_replicas = 0  # Scale to zero when not in use
    max_replicas = 3  # Scale up to 3 instances under load
  }

  ingress {
    external_enabled = true
    target_port      = 8501
    traffic_weight {
      percentage = 100
      latest_revision = true
    }
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.main.admin_password
  }

  tags = {
    Environment = var.environment
    Project     = "SummarEaseAI"
    Service     = "Frontend"
  }
} 