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
    docker = {
      source  = "kreuzwerker/docker"
      version = "~>3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

provider "docker" {
  registry_auth {
    address  = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password = azurerm_container_registry.main.admin_password
  }
} 