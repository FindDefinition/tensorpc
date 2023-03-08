def get_url_port(url: str):
    port = -1
    # TODO if url is ipv6?
    url_parts = url.split(":")
    if len(url_parts) == 1 or len(url_parts) == 8:
        return url, port
    elif len(url_parts) == 2:
        return url_parts[0], int(url_parts[1])
    return ":".join(url_parts[:-1]), int(url_parts[-1])


def convert_url_to_local(url: str):
    url_noport, url_port = get_url_port(url)
    if url_port == -1:
        return "localhost"
    return f"localhost:{url_port}"
