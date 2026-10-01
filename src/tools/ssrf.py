import asyncio
import ipaddress
import socket

from core.errors import AppError, ErrorCode

type IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def is_public(ip: IpAddress) -> bool:
    """False for private, loopback, link-local, unique-local (fc00::/7), reserved and similar ranges."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 is just 127.0.0.1
    return ip.is_global


async def _lookup(host: str, port: int) -> list[IpAddress]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [ipaddress.ip_address(str(info[4][0]).split("%")[0]) for info in infos]


async def resolve_public(host: str, port: int) -> list[str]:
    """Resolves `host` and returns its addresses (IPv4 first); refuses the host if any address is not public.

    Callers must connect to the returned IP, not the name, so a second DNS answer can't swap in a private one.
    """
    try:
        addresses = [ipaddress.ip_address(host.strip("[]"))]
    except ValueError:
        try:
            addresses = await _lookup(host, port)
        except OSError:
            raise AppError(ErrorCode.URL_FETCH_FAILED, "That website could not be found.") from None

    if not addresses:
        raise AppError(ErrorCode.URL_FETCH_FAILED, "That website could not be found.")
    if not all(is_public(address) for address in addresses):
        raise AppError(ErrorCode.URL_BLOCKED, "That address is not allowed.")
    ordered = sorted(dict.fromkeys(addresses), key=lambda a: a.version)
    return [str(address) for address in ordered]
