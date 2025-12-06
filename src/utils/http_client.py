import requests

class HttpClient:
    """
    Quản lý session và gửi request GET đến API tìm kiếm.
    """
    def __init__(self):
        self.session = requests.Session()
        # Đặt User-Agent thân thiện cho mục đích nghiên cứu
        self.session.headers.update({
            'User-Agent': 'Q-Learning SQLi Agent (Academic Research)'
        })

    def send_search_query(self, url, query, param_name='q'):
        """
        Gửi payload 'query' đến URL mục tiêu.
        """
        # Allow configurable param name (default 'q')
        params = {param_name: query}
        try:
            # Đặt timeout để tránh agent bị "treo"
            response = self.session.get(url, params=params, timeout=5)
            return response
        except requests.RequestException as e:
            print(f"[HttpClient Error]: {e}")
            return None