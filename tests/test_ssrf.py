import ipaddress

import pytest

from core.errors import AppError, ErrorCode
from tools import ssrf
from tools.ssrf import IpAddress, is_public, resolve_public


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.1.1",
        "127.0.0.1",
        "127.1.2.3",
        "169.254.169.254",
        "0.0.0.0",
        "100.64.0.1",
        "::1",
        "fc00::1",
        "fd12:3456::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.5",
    ],
)
def test_non_public_addresses_are_refused(address: str) -> None:
    assert not is_public(ipaddress.ip_address(address))


@pytest.mark.parametrize(
    "address",
    ["8.8.8.8", "93.184.216.34", "172.15.255.255", "172.32.0.1", "192.169.0.1", "2606:4700:4700::1111"],
)
def test_public_addresses_are_allowed(address: str) -> None:
    assert is_public(ipaddress.ip_address(address))


def fake_dns(monkeypatch: pytest.MonkeyPatch, answers: list[str] | Exception) -> None:
    async def lookup(host: str, port: int) -> list[IpAddress]:
        if isinstance(answers, Exception):
            raise answers
        return [ipaddress.ip_address(a) for a in answers]

    monkeypatch.setattr(ssrf, "_lookup", lookup)


async def test_ip_literals_are_checked_without_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dns(monkeypatch, OSError("DNS must not be used for literals"))

    with pytest.raises(AppError) as exc:
        await resolve_public("169.254.169.254", 80)
    assert exc.value.code is ErrorCode.URL_BLOCKED

    with pytest.raises(AppError) as exc:
        await resolve_public("[::1]", 80)
    assert exc.value.code is ErrorCode.URL_BLOCKED

    assert await resolve_public("8.8.8.8", 80) == ["8.8.8.8"]


async def test_hostname_resolving_to_a_private_address_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dns(monkeypatch, ["192.168.0.10"])

    with pytest.raises(AppError) as exc:
        await resolve_public("intranet.example.com", 443)

    assert exc.value.code is ErrorCode.URL_BLOCKED


async def test_one_private_answer_blocks_the_whole_host(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dns(monkeypatch, ["93.184.216.34", "10.0.0.1"])

    with pytest.raises(AppError) as exc:
        await resolve_public("mixed.example.com", 443)

    assert exc.value.code is ErrorCode.URL_BLOCKED


async def test_public_addresses_come_back_ipv4_first(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dns(monkeypatch, ["2606:4700:4700::1111", "93.184.216.34", "93.184.216.34"])

    assert await resolve_public("example.com", 443) == ["93.184.216.34", "2606:4700:4700::1111"]


async def test_unresolvable_host_is_a_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dns(monkeypatch, OSError("no such host"))

    with pytest.raises(AppError) as exc:
        await resolve_public("nope.invalid", 443)

    assert exc.value.code is ErrorCode.URL_FETCH_FAILED
