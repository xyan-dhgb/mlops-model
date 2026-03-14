locals {
  repositories = {
    preprocessing = {
      name        = "${var.project_name}-preprocessing"
      description = "Data preprocessing & feature engineering for ${var.project_name}"
    }
    training = {
      name        = "${var.project_name}-training"
      description = "TensorFlow/Keras model training for ${var.project_name}"
    }
    serving = {
      name        = "${var.project_name}-serving"
      description = "TensorFlow Serving / inference API for ${var.project_name}"
    }
  }
}

resource "aws_ecr_repository" "this" {
  for_each = local.repositories

  name                 = "${var.environment}/${each.value.name}"
  image_tag_mutability = var.image_tag_mutability

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(var.tags, {
    Name        = each.value.name
    Description = each.value.description
    Environment = var.environment
    ManagedBy   = "terraform"
    Framework   = "tensorflow"
  })
}

# Lifecycle policy — expire untagged in 1 day, keep last N tagged
resource "aws_ecr_lifecycle_policy" "this" {
  for_each   = aws_ecr_repository.this
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep last ${var.max_image_count} tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "sha-", "dev-", "prod-"]
          countType     = "imageCountMoreThan"
          countNumber   = var.max_image_count
        }
        action = { type = "expire" }
      }
    ]
  })
}

# Optional cross-account pull policy
resource "aws_ecr_repository_policy" "this" {
  for_each   = var.allowed_account_ids != null ? aws_ecr_repository.this : {}
  repository = each.value.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCrossAccountPull"
        Effect = "Allow"
        Principal = {
          AWS = [for id in var.allowed_account_ids : "arn:aws:iam::${id}:root"]
        }
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
      }
    ]
  })
}
