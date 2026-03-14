output "repository_urls" {
  description = "Map of repository key → full ECR URL"
  value = {
    for k, repo in aws_ecr_repository.this : k => repo.repository_url
  }
}

output "repository_arns" {
  description = "Map of repository key → ECR ARN"
  value = {
    for k, repo in aws_ecr_repository.this : k => repo.arn
  }
}

output "registry_id" {
  description = "AWS Account ID used as the ECR registry ID"
  value       = values(aws_ecr_repository.this)[0].registry_id
}

# ── Convenience outputs per image ────────────────────────────────────────────

output "preprocessing_repository_url" {
  description = "Full ECR URL for the preprocessing image"
  value       = aws_ecr_repository.this["preprocessing"].repository_url
}

output "training_repository_url" {
  description = "Full ECR URL for the TensorFlow training image"
  value       = aws_ecr_repository.this["training"].repository_url
}

output "serving_repository_url" {
  description = "Full ECR URL for the TensorFlow Serving inference image"
  value       = aws_ecr_repository.this["serving"].repository_url
}
