# Docker Provider for Image Building and Pushing
terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

# Configure Docker provider
provider "docker" {
  registry_auth {
    address  = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password = azurerm_container_registry.main.admin_password
  }
}

# Build and push backend image
resource "docker_image" "backend" {
  name = "${azurerm_container_registry.main.login_server}/summarease-backend:latest"
  
  build {
    context    = ".."  # Build from project root
    dockerfile = "backend/Dockerfile"
    
    tag = [
      "${azurerm_container_registry.main.login_server}/summarease-backend:latest",
      "${azurerm_container_registry.main.login_server}/summarease-backend:${formatdate("YYYYMMDD-hhmm", timestamp())}"
    ]
  }
  
  triggers = {
    dir_sha1 = sha1(join("", [
      for f in fileset("../backend", "**") : filesha1("../backend/${f}")
    ]))
  }
}

# Build and push frontend image
resource "docker_image" "frontend" {
  name = "${azurerm_container_registry.main.login_server}/summarease-frontend:latest"
  
  build {
    context    = ".."  # Build from project root
    dockerfile = "frontend/Dockerfile"
    
    tag = [
      "${azurerm_container_registry.main.login_server}/summarease-frontend:latest",
      "${azurerm_container_registry.main.login_server}/summarease-frontend:${formatdate("YYYYMMDD-hhmm", timestamp())}"
    ]
  }
  
  triggers = {
    dir_sha1 = sha1(join("", [
      for f in fileset("../frontend", "**") : filesha1("../frontend/${f}")
    ]))
  }
}

# Push backend image to ACR
resource "docker_registry_image" "backend" {
  name = docker_image.backend.name
  
  depends_on = [
    azurerm_container_registry.main,
    docker_image.backend
  ]
}

# Push frontend image to ACR
resource "docker_registry_image" "frontend" {
  name = docker_image.frontend.name
  
  depends_on = [
    azurerm_container_registry.main,
    docker_image.frontend
  ]
} 