#!/bin/bash

# 🧠 Omeruta Brain Complete System Starter
# This script starts all services needed for the real-time AI system

echo "🚀 Starting Omeruta Brain Complete System..."
echo "================================================"

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Redis is running
echo -e "${BLUE}Checking Redis...${NC}"
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis is not running. Please start Redis first:${NC}"
    echo "   brew services start redis  # macOS"
    echo "   sudo systemctl start redis  # Linux"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run from omeruta_brain directory.${NC}"
    exit 1
fi

echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if all dependencies are installed
echo -e "${BLUE}Checking dependencies...${NC}"
python -c "import channels, celery, daphne" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Some dependencies missing. Installing...${NC}"
    pip install channels channels-redis daphne celery[redis] django-celery-beat django-celery-results
fi

echo ""
echo -e "${GREEN}🎉 System Status:${NC}"
echo -e "${GREEN}✅ Redis: Running${NC}"
echo -e "${GREEN}✅ Dependencies: Installed${NC}"
echo -e "${GREEN}✅ Ready to start all services${NC}"
echo ""

echo -e "${YELLOW}Choose your startup mode:${NC}"
echo "1) 🚀 Full System (WebSocket + Celery + Background tasks)"
echo "2) 🌐 WebSocket Server Only (for testing WebSocket demo)"
echo "3) 🔧 Celery Worker Only (for async processing)"
echo "4) 📊 System Status Check"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo -e "${BLUE}Starting Full System...${NC}"
        echo ""
        echo -e "${YELLOW}This will open multiple terminal tabs/windows:${NC}"
        echo "1. WebSocket Server (Daphne)"
        echo "2. Celery Worker (AI Processing)"
        echo "3. Celery Beat (Background Tasks)"
        echo ""
        
        # For macOS, open new Terminal tabs
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo -e "${BLUE}Opening Terminal tabs...${NC}"
            
            # Start WebSocket server in new tab
            osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && source venv/bin/activate && echo '🌐 Starting WebSocket Server...' && python manage.py start_websocket_server\""
            
            # Start Celery worker in new tab
            osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && source venv/bin/activate && echo '🤖 Starting Celery Worker...' && python manage.py start_celery_worker --queue ai_high_priority,ai_processing\""
            
            # Start Celery beat in new tab
            osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && source venv/bin/activate && echo '⏰ Starting Celery Beat...' && python manage.py start_celery_beat\""
            
        else
            echo -e "${YELLOW}Please run these commands in separate terminals:${NC}"
            echo ""
            echo -e "${BLUE}Terminal 1 - WebSocket Server:${NC}"
            echo "cd $(pwd) && source venv/bin/activate && python manage.py start_websocket_server"
            echo ""
            echo -e "${BLUE}Terminal 2 - Celery Worker:${NC}"
            echo "cd $(pwd) && source venv/bin/activate && python manage.py start_celery_worker --queue ai_high_priority,ai_processing"
            echo ""
            echo -e "${BLUE}Terminal 3 - Celery Beat:${NC}"
            echo "cd $(pwd) && source venv/bin/activate && python manage.py start_celery_beat"
        fi
        
        echo ""
        echo -e "${GREEN}🎯 Once all services are running:${NC}"
        echo -e "${BLUE}Demo URL: http://localhost:8000/static/websocket-demo.html${NC}"
        ;;
        
    2)
        echo -e "${BLUE}Starting WebSocket Server Only...${NC}"
        python manage.py start_websocket_server
        ;;
        
    3)
        echo -e "${BLUE}Starting Celery Worker Only...${NC}"
        python manage.py start_celery_worker --queue ai_high_priority,ai_processing --verbosity 2
        ;;
        
    4)
        echo -e "${BLUE}System Status Check...${NC}"
        echo ""
        
        # Check Django
        echo -e "${BLUE}Django Status:${NC}"
        python manage.py check --verbosity 0
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Django: OK${NC}"
        else
            echo -e "${RED}❌ Django: Issues found${NC}"
        fi
        
        # Check database
        echo -e "${BLUE}Database Status:${NC}"
        python manage.py showmigrations --verbosity 0 | grep -q "\[ \]"
        if [ $? -eq 1 ]; then
            echo -e "${GREEN}✅ Database: All migrations applied${NC}"
        else
            echo -e "${YELLOW}⚠️  Database: Pending migrations${NC}"
            echo "Run: python manage.py migrate"
        fi
        
        # Check TinyLlama
        echo -e "${BLUE}AI Model Status:${NC}"
        python manage.py test_tinyllama --verbosity 0 > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ TinyLlama: Ready${NC}"
        else
            echo -e "${YELLOW}⚠️  TinyLlama: May have issues (will fallback gracefully)${NC}"
        fi
        
        echo ""
        echo -e "${GREEN}🎯 System Ready! Choose option 1 or 2 to start services.${NC}"
        ;;
        
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🧠 Omeruta Brain System Manager${NC}"
echo -e "${BLUE}For help: cat WEBSOCKET_CELERY_GUIDE.md${NC}" 