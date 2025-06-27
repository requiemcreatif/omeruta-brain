# Omeruta Brain - Django Backend

AI-powered brain and knowledge management system backend built with Django and PostgreSQL.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Virtual environment

### Installation

1. **Clone and navigate to the backend directory**

```bash
cd omeruta_brain
```

2. **Activate virtual environment**

```bash
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

- Copy `.env.example` to `.env` if needed
- Update database credentials and API keys in `.env`

5. **Set up database**

```bash
# Make sure PostgreSQL is running
createdb omeruta_brain

# Run migrations
python manage.py migrate
```

6. **Create superuser (optional)**

```bash
python manage.py createsuperuser
```

_Note: A superuser with email `requiemcreatif@gmail.com` is already configured_

7. **Start the server**

```bash
# Method 1: Using the startup script
python start_api.py

# Method 2: Using Django directly
python manage.py runserver 0.0.0.0:8000
```

## 📡 API Endpoints

### Core Endpoints

- `GET /api/health/` - Health check
- `GET /api/info/` - API information

### Authentication Endpoints

- `POST /api/auth/register/` - User registration
- `POST /api/auth/login/` - User login
- `POST /api/auth/logout/` - User logout
- `GET/PUT /api/auth/profile/` - User profile management
- `POST /api/auth/change-password/` - Change password
- `GET /api/auth/check-auth/` - Check authentication status
- `POST /api/auth/token/refresh/` - Refresh JWT token

### Admin Endpoints

- `POST /api/auth/admin/login/` - Admin login
- `GET /api/auth/admin/dashboard/` - Admin dashboard data

### Django Admin

- `/admin/` - Django admin panel

## 🔐 Authentication

The API uses JWT (JSON Web Token) authentication with the following features:

- **Access tokens**: 60 minutes lifetime
- **Refresh tokens**: 7 days lifetime
- **Token rotation**: Refresh tokens are rotated on use
- **Blacklisting**: Old tokens are blacklisted for security

### Authentication Header

```
Authorization: Bearer <access_token>
```

### Login Response Example

```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "is_staff": false
  },
  "tokens": {
    "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  },
  "message": "Login successful"
}
```

## 🏗️ Project Structure

```
omeruta_brain/
├── omeruta_brain_project/     # Django project settings
│   ├── __init__.py
│   ├── settings.py           # Main settings
│   ├── urls.py              # URL routing
│   └── wsgi.py
├── core/                    # Core app
│   ├── views.py            # Basic API views
│   └── urls.py
├── authentication/         # Authentication app
│   ├── serializers.py     # User serializers
│   ├── views.py          # Auth views
│   └── urls.py
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── manage.py            # Django management
└── start_api.py        # Server startup script
```

## 🔧 Configuration

### Environment Variables

The following environment variables can be configured in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/omeruta_brain
DB_NAME=omeruta_brain
DB_USER=omeruta_user
DB_PASSWORD=pebzyn-1gUvfo-dawzip
DB_HOST=localhost
DB_PORT=5432

# Django
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# JWT
JWT_SECRET_KEY=your-jwt-secret
JWT_ACCESS_TOKEN_LIFETIME=60
JWT_REFRESH_TOKEN_LIFETIME=7

# AI APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### CORS Configuration

The API is configured to accept requests from:

- `http://localhost:3000` (Next.js frontend)
- `http://127.0.0.1:3000`

## 🧪 Testing

### API Testing with curl

**Health Check**

```bash
curl http://localhost:8000/api/health/
```

**User Registration**

```bash
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "securepassword123",
    "password_confirm": "securepassword123",
    "first_name": "Test",
    "last_name": "User"
  }'
```

**User Login**

```bash
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "securepassword123"
  }'
```

**Admin Login**

```bash
curl -X POST http://localhost:8000/api/auth/admin/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "requiemcreatif@gmail.com",
    "password": "fucxaj-kymvYk-rakwa4"
  }'
```

## 🔄 Development Workflow

1. **Start the server**

```bash
source venv/bin/activate
python start_api.py
```

2. **Make model changes**

```bash
python manage.py makemigrations
python manage.py migrate
```

3. **Create new apps**

```bash
python manage.py startapp app_name
```

4. **Access admin panel**

- Go to `http://localhost:8000/admin/`
- Login with: `requiemcreatif@gmail.com` / `fucxaj-kymvYk-rakwa4`

## 🚨 Superuser Credentials

**Email**: `requiemcreatif@gmail.com`  
**Password**: `fucxaj-kymvYk-rakwa4`

_Note: Change these credentials in production!_

## 🔗 Frontend Integration

This backend is designed to work with the Next.js frontend in the `omeruta-brain-dashboard` directory. The API provides:

- CORS-enabled endpoints for web requests
- JWT token-based authentication
- RESTful API design
- Comprehensive error handling
- Admin authentication for dashboard access

## 📝 Next Steps

1. Add more apps for specific features (knowledge management, AI processing, etc.)
2. Implement additional authentication methods (OAuth, social auth)
3. Add API documentation with Swagger/OpenAPI
4. Set up production deployment configuration
5. Add comprehensive testing suite
6. Implement caching with Redis
7. Add Celery for background tasks
# omeruta-brain
