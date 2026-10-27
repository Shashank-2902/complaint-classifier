variable "location" {
  description = "Azure region for every resource."
  type        = string
  default     = "germanywestcentral"
}

variable "name_prefix" {
  description = "Short lowercase prefix for resource names (letters and digits only)."
  type        = string
  default     = "complaintclf"

  validation {
    condition     = can(regex("^[a-z0-9]{3,20}$", var.name_prefix))
    error_message = "name_prefix must be 3 to 20 lowercase letters or digits."
  }
}

variable "image_tag" {
  description = "Tag of the image in the registry that the container app runs."
  type        = string
  default     = "latest"
}

variable "tags" {
  description = "Tags put on every resource."
  type        = map(string)
  default = {
    project = "complaint-classifier"
    managed = "terraform"
  }
}
