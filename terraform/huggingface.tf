# Configure the Hugging Face provider
terraform {
  required_providers {
    huggingface = {
      source = "huggingface/huggingface"
    }
  }
}

# Hugging Face Space resource
resource "huggingface_space" "backend" {
  name        = "${var.project_name}-backend"
  organization = var.huggingface_org  # Your HF username or organization
  type        = "docker"
  hardware    = "cpu-basic"

  # Space configuration
  container_config {
    image = "${var.docker_registry}/${var.project_name}-backend:latest"
    port  = 7860

    env = {
      FLASK_ENV = "production"
      PYTHONPATH = "/app"
      FLASK_APP = "backend/api.py"
    }

    secrets = {
      OPENAI_API_KEY = var.openai_api_key
    }
  }
}

# Variables for Hugging Face deployment
variable "huggingface_org" {
  description = "Hugging Face organization or username"
  type        = string
}

variable "docker_registry" {
  description = "Docker registry URL"
  type        = string
  default     = "hf.co"  # Hugging Face's registry
}

# Output the Space URL
output "huggingface_space_url" {
  value = "https://huggingface.co/spaces/${var.huggingface_org}/${huggingface_space.backend.name}"
} 