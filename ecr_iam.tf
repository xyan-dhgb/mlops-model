# =============================================================
# configs/terraform/ecr_iam.tf
# Tạo ECR repositories + IAM Role cho GitHub Actions OIDC
# Chạy: terraform init && terraform apply
# =============================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "cancer-mlops-terraform-state"
    key    = "ecr/terraform.tfstate"
    region = "ap-southeast-1"
  }
}

provider "aws" {
  region = "ap-southeast-1"
}

# ── Variables ─────────────────────────────────────────────
variable "github_org"  { default = "your-github-org" }
variable "github_repo" { default = "cancer-mlops" }
variable "project_name" { default = "cancer-mlops" }

# ── Data: AWS Account ID ──────────────────────────────────
data "aws_caller_identity" "current" {}

# =============================================================
# ECR Repositories
# =============================================================
resource "aws_ecr_repository" "preprocessing" {
  name                 = "${var.project_name}/preprocessing"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true   # Tự động scan khi push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Project   = var.project_name
    Component = "preprocessing"
    ManagedBy = "terraform"
  }
}

resource "aws_ecr_repository" "training" {
  name                 = "${var.project_name}/training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Project   = var.project_name
    Component = "training"
    ManagedBy = "terraform"
  }
}

resource "aws_ecr_repository" "serving" {
  name                 = "${var.project_name}/serving"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Project   = var.project_name
    Component = "serving"
    ManagedBy = "terraform"
  }
}

# ── Lifecycle Policy: Giữ tối đa 10 images ────────────────
resource "aws_ecr_lifecycle_policy" "keep_last_10" {
  for_each   = {
    preprocessing = aws_ecr_repository.preprocessing.name
    training      = aws_ecr_repository.training.name
    serving       = aws_ecr_repository.serving.name
  }
  repository = each.value

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images tagged with semver"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep last 5 dev images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["dev-"]
          countType     = "imageCountMoreThan"
          countNumber   = 5
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 3
        description  = "Delete untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      }
    ]
  })
}

# =============================================================
# OIDC Provider cho GitHub Actions (không cần long-lived keys)
# =============================================================
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

# Nếu OIDC provider chưa có → tạo mới
resource "aws_iam_openid_connect_provider" "github_actions" {
  count = length(data.aws_iam_openid_connect_provider.github.arn) == 0 ? 1 : 0

  url = "https://token.actions.githubusercontent.com"

  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# =============================================================
# IAM Role cho GitHub Actions
# =============================================================
resource "aws_iam_role" "github_actions_ecr" {
  name = "github-actions-ecr-push-${var.project_name}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            # Chỉ cho phép từ repo cụ thể, mọi branch
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_org}/${var.github_repo}:*"
          }
        }
      }
    ]
  })

  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
  }
}

# ── ECR Push/Pull Permissions ─────────────────────────────
resource "aws_iam_role_policy" "ecr_push_pull" {
  name = "ecr-push-pull-policy"
  role = aws_iam_role.github_actions_ecr.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRGetAuthToken"
        Effect = "Allow"
        Action = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPushPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeRepositories",
          "ecr:CreateRepository",
          "ecr:DescribeImages",
          "ecr:ListImages",
          "ecr:BatchDeleteImage",
          "ecr:GetLifecyclePolicy",
          "ecr:PutLifecyclePolicy",
        ]
        Resource = [
          aws_ecr_repository.preprocessing.arn,
          aws_ecr_repository.training.arn,
          aws_ecr_repository.serving.arn,
        ]
      }
    ]
  })
}

# =============================================================
# Outputs (dùng cho GitHub Secrets)
# =============================================================
output "role_arn" {
  value       = aws_iam_role.github_actions_ecr.arn
  description = "Dùng giá trị này cho GitHub Secret: AWS_IAM_ROLE_ARN"
}

output "ecr_registry" {
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.ap-southeast-1.amazonaws.com"
  description = "ECR Registry URL"
}

output "preprocessing_repo_url" {
  value = aws_ecr_repository.preprocessing.repository_url
}

output "training_repo_url" {
  value = aws_ecr_repository.training.repository_url
}

output "serving_repo_url" {
  value = aws_ecr_repository.serving.repository_url
}
