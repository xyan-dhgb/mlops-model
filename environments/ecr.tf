# environments/dev/ecr.tf
# ─────────────────────────────────────────────────────────────────────────────
# Drop this file into environments/dev/ alongside your existing main.tf.
# It provisions 3 ECR repositories: preprocessing, training, serving.
# ─────────────────────────────────────────────────────────────────────────────

module "ecr" {
  source = "../../modules/ecr"

  project_name         = var.project_name   # e.g. "mlops" — add to variables.tf if missing
  environment          = var.environment    # "dev"
  image_tag_mutability = "MUTABLE"          # dev: allow tag overwrites
  scan_on_push         = true
  max_image_count      = 10

  # Uncomment to allow another AWS account to pull images:
  # allowed_account_ids = ["123456789012"]

  tags = {
    Project    = var.project_name
    Team       = "mlops"
    Framework  = "tensorflow"
    CostCenter = "ml-infra"
  }
}

# ── Outputs — copy these values into GitHub Actions → Repo Variables ──────────

output "ecr_registry_id" {
  description = "ECR registry ID (= AWS account ID)"
  value       = module.ecr.registry_id
}

output "ecr_preprocessing_url" {
  description = "ECR URL for preprocessing image"
  value       = module.ecr.preprocessing_repository_url
}

output "ecr_training_url" {
  description = "ECR URL for TensorFlow training image"
  value       = module.ecr.training_repository_url
}

output "ecr_serving_url" {
  description = "ECR URL for TensorFlow Serving image"
  value       = module.ecr.serving_repository_url
}
