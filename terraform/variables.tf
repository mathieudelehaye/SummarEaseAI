variable "resource_group_name" {
  description = "Name of the Azure Resource Group"
  type        = string
  default     = "rg-summarease-ai"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "West Europe"
}

variable "environment" {
  description = "Environment name (production, staging)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "summarease"
}

variable "acr_name" {
  description = "Azure Container Registry name (must be globally unique)"
  type        = string
  default     = "acrsummarease"
}

# Sensitive variables - these should be provided via terraform.tfvars
variable "openai_api_key" {
  description = "OpenAI API key for AI services"
  type        = string
  sensitive   = true
}

variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Azure tenant ID"
  type        = string
  sensitive   = true
}

variable "client_id" {
  description = "Azure service principal client ID"
  type        = string
  sensitive   = true
}

variable "client_secret" {
  description = "Azure service principal client secret"
  type        = string
  sensitive   = true
}

variable "user_object_id" {
  description = "Object ID of the user who needs access to Key Vault"
  type        = string
  sensitive   = true
}

variable "existing_container_env_name" {
  description = "Name of the existing Container App Environment"
  type        = string
  default     = "calorie-tracker-env"
}

variable "existing_container_env_rg" {
  description = "Resource Group of the existing Container App Environment"
  type        = string
  default     = "calorie-tracker-rg"
} 