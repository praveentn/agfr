# scripts/test_agents.py
import requests
import time
import sys

def test_agent_connectivity():
    """Test connectivity to all MCP agent servers"""
    agents = [
        ('Web Search', 'http://localhost:9101'),
        ('Tabulator', 'http://localhost:9102'), 
        ('NLP Summarizer', 'http://localhost:9103'),
        ('Calculator', 'http://localhost:9104')
    ]
    
    print("Testing agent connectivity...")
    all_online = True
    
    for name, url in agents:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code in [200, 404]:  # 404 is ok for basic health check
                print(f'✓ {name} agent: Online')
            else:
                print(f'⚠ {name} agent: Responding but status {response.status_code}')
                all_online = False
        except requests.exceptions.ConnectRefused:
            print(f'✗ {name} agent: Connection refused (not started)')
            all_online = False
        except requests.exceptions.Timeout:
            print(f'⚠ {name} agent: Timeout (slow response)')
            all_online = False
        except Exception as e:
            print(f'⚠ {name} agent: Error - {str(e)[:50]}')
            all_online = False
    
    if all_online:
        print("\n✓ All agents are responding!")
    else:
        print("\n⚠ Some agents may still be starting up...")
    
    return all_online

def validate_configuration():
    """Validate system configuration"""
    print("\nValidating configuration...")
    
    try:
        from agentic.core.config import settings
        print('✓ Configuration loaded successfully')
        
        # Check Azure OpenAI
        if settings.azure_openai_endpoint and settings.azure_openai_api_key:
            endpoint = settings.azure_openai_endpoint
            print(f'✓ Azure OpenAI configured: {endpoint[:50]}...')
        else:
            print('⚠ Azure OpenAI not configured (will use fallback modes)')
        
        print(f'✓ Auth Token: {settings.auth_token}')
        print(f'✓ Database: {settings.database_url}')
        
        # Check directories
        import os
        dirs_to_check = [
            ('agents', settings.agents_dir),
            ('workflows', settings.workflows_dir),
            ('prompts', settings.prompts_dir)
        ]
        
        for name, path in dirs_to_check:
            if os.path.exists(path):
                print(f'✓ {name} directory: {path}')
            else:
                print(f'⚠ {name} directory: {path} (will be created)')
        
        return True
        
    except Exception as e:
        print(f'✗ Configuration issue: {e}')
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "connectivity":
            test_agent_connectivity()
        elif sys.argv[1] == "config":
            validate_configuration()
        elif sys.argv[1] == "all":
            validate_configuration()
            print("\n" + "="*50)
            test_agent_connectivity()
    else:
        validate_configuration()
        print("\n" + "="*50)
        test_agent_connectivity()

