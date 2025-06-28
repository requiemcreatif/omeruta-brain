#!/usr/bin/env python
"""
Omeruta Brain API Server Startup Script
"""
import os
import sys
import django
from django.core.management import execute_from_command_line

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "omeruta_brain_project.settings")

    # Setup Django
    django.setup()

    print("🧠 Starting Omeruta Brain API Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔒 Admin panel available at: http://localhost:8000/admin/")
    print("📚 API endpoints:")
    print("   - API info:     http://localhost:8000/core/info/")
    print("   - Health check: http://localhost:8000/core/health/")
    print("   - Admin:        http://localhost:8000/admin/")
    print("   - Swagger UI:   http://localhost:8000/api/docs/")
    print("\n🚀 Starting server...")

    # Start the development server
    execute_from_command_line(["manage.py", "runserver", "0.0.0.0:8000"])
