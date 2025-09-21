def oauth_debug():
    st.set_page_config(page_title="OAuth Debug Tool", page_icon="🔧")
    
    st.title("🔧 OAuth Debug Tool")
    
    # Show current configuration
    st.markdown("### 📋 Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**App Configuration:**")
        st.code(f"""
Client ID: {CONFIG.get('GOOGLE_CLIENT_ID', 'MISSING')[:30]}...
App Domain: {CONFIG.get('APP_DOMAIN', 'MISSING')}
Redirect URI: {REDIRECT_URI}
Authorized Email: {CONFIG.get('AUTHORIZED_EMAIL', 'MISSING')}
        """)
    
    with col2:
        st.markdown("**Current Request Info:**")
        
        # Get current URL info
        query_params = dict(st.query_params)
        
        st.code(f"""
Current URL: {st.context.request.url if hasattr(st.context, 'request') else 'Unknown'}
Query Params: {query_params}
        """)
    
    # Show what to add to Google OAuth
    st.markdown("### 🔗 Google OAuth Configuration")
    st.info("Add BOTH of these URLs to your Google OAuth Client:")
    
    st.code(f"""
Authorized redirect URIs:
1. {REDIRECT_URI}
2. {REDIRECT_URI}/
    """)
    
    # Test OAuth flow
    st.markdown("### 🧪 Test OAuth Flow")
    
    # Check for OAuth response
    if 'code' in st.query_params:
        st.success("✅ OAuth callback received!")
        code = st.query_params['code']
        st.write(f"**Authorization Code**: {code[:50]}...")
        
        # Test token exchange
        if st.button("🔄 Test Token Exchange"):
            with st.spinner("Testing token exchange..."):
                token_data = {
                    'client_id': CONFIG['GOOGLE_CLIENT_ID'],
                    'client_secret': CONFIG['GOOGLE_CLIENT_SECRET'],
                    'code': code,
                    'grant_type': 'authorization_code',
                    'redirect_uri': REDIRECT_URI,
                }
                
                try:
                    response = requests.post('https://oauth2.googleapis.com/token', data=token_data)
                    st.write(f"**Response Status**: {response.status_code}")
                    
                    if response.status_code == 200:
                        token_response = response.json()
                        st.success("✅ Token exchange successful!")
                        
                        if 'id_token' in token_response:
                            # Decode ID token
                            id_token = token_response['id_token']
                            payload_part = id_token.split('.')[1]
                            payload_part += '=' * (4 - len(payload_part) % 4)
                            payload = json.loads(base64.urlsafe_b64decode(payload_part))
                            
                            st.write(f"**User Email**: {payload.get('email')}")
                            st.write(f"**Authorized Email**: {CONFIG['AUTHORIZED_EMAIL']}")
                            
                            if payload.get('email') == CONFIG['AUTHORIZED_EMAIL']:
                                st.success("✅ Email authorization successful!")
                            else:
                                st.error("❌ Email not authorized")
                        else:
                            st.error("❌ No ID token in response")
                            st.json(token_response)
                    else:
                        st.error(f"❌ Token exchange failed: {response.status_code}")
                        st.json(response.json())
                        
                except Exception as e:
                    st.error(f"❌ Token exchange error: {e}")
    
    elif 'error' in st.query_params:
        st.error(f"❌ OAuth Error: {st.query_params['error']}")
        if 'error_description' in st.query_params:
            st.write(f"**Description**: {st.query_params['error_description']}")
        
        st.markdown("### 🔍 Common Solutions:")
        st.markdown("""
        - **403 Forbidden**: Check OAuth consent screen test users
        - **redirect_uri_mismatch**: Update Google OAuth redirect URIs
        - **unauthorized_client**: Check client ID and secret
        """)
    
    else:
        # Show OAuth URL
        if CONFIG.get('GOOGLE_CLIENT_ID'):
            params = {
                'client_id': CONFIG['GOOGLE_CLIENT_ID'],
                'redirect_uri': REDIRECT_URI,
                'scope': 'openid email profile',
                'response_type': 'code',
                'access_type': 'offline',
                'prompt': 'consent',
                'include_granted_scopes': 'true'
            }
            
            auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
            
            st.markdown("### 🚀 Test OAuth Flow")
            
            # Show the auth URL for manual inspection
            with st.expander("🔍 Inspect OAuth URL"):
                st.code(auth_url)
            
            st.markdown(f"""
            <a href="{auth_url}" target="_self">
                <button style="
                    background-color: #4285f4;
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                    text-decoration: none;
                    display: inline-block;
                ">
                    🔐 Test Google OAuth
                </button>
            </a>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Google Client ID not configured")
    
    # Configuration checklist
    st.markdown("### ✅ Configuration Checklist")
    
    checklist = [
        ("Google Client ID configured", bool(CONFIG.get('GOOGLE_CLIENT_ID'))),
        ("Google Client Secret configured", bool(CONFIG.get('GOOGLE_CLIENT_SECRET'))),
        ("Authorized email configured", bool(CONFIG.get('AUTHORIZED_EMAIL'))),
        ("App domain configured", bool(CONFIG.get('APP_DOMAIN'))),
    ]
    
    for item, status in checklist:
        if status:
            st.success(f"✅ {item}")
        else:
            st.error(f"❌ {item}")