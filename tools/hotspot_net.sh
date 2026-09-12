#!/bin/bash
# 아이폰 핫스팟용 IPv4 수동 설정 ↔ 원래 DHCP 를 오가는 토글.
#
# 왜 필요한가
#   아이폰 핫스팟은 애플 기기(맥북)에는 IPv6 전용 주소만 주고, ESP32 같은
#   일반 기기에는 IPv4(172.20.10.x)를 준다. 그래서 맥북만 IPv4 가 없어
#   서로 통신이 안 된다. 맥북에 IPv4 를 수동으로 박아야 한다.
#
#   그런데 `networksetup -setmanual` 은 **네트워크별이 아니라 Wi-Fi 서비스
#   전체**에 적용된다. 학교 Wi-Fi 로 돌아가도 수동 IP 가 남아 인터넷이 죽고,
#   DNS 까지 손대면 복구가 번거롭다. 이 스크립트가 양쪽을 한 번에 되돌린다.
#
# 사용법
#   tools/hotspot_net.sh on     # 핫스팟용 수동 IP (기본 172.20.10.8)
#   tools/hotspot_net.sh off    # 원래대로 (DHCP + DNS 자동)
#   tools/hotspot_net.sh status
#
#   IP 를 바꾸려면: tools/hotspot_net.sh on 172.20.10.9
#
# 주의: app_config.h 의 SERVER_IPV4 와 같은 값이어야 ESP32 가 찾아온다.

set -euo pipefail

SERVICE="Wi-Fi"
IP="${2:-172.20.10.8}"
MASK="255.255.255.240"
ROUTER="172.20.10.1"

case "${1:-status}" in
  on)
    networksetup -setmanual "$SERVICE" "$IP" "$MASK" "$ROUTER"
    # 핫스팟 게이트웨이를 DNS 로. 안 넣으면 이름 해석이 전부 막힌다.
    networksetup -setdnsservers "$SERVICE" "$ROUTER"
    sleep 2
    echo "핫스팟 모드: $(ipconfig getifaddr en0 2>/dev/null || echo '설정 반영 대기')"
    echo "  → app_config.h 의 SERVER_IPV4 가 $IP 인지 확인하세요"
    ;;
  off)
    networksetup -setdhcp "$SERVICE"
    networksetup -setdnsservers "$SERVICE" "Empty"   # DHCP 가 주는 DNS 로 복귀
    sleep 2
    echo "원래대로: $(ipconfig getifaddr en0 2>/dev/null || echo '주소 대기 중')"
    ;;
  status)
    networksetup -getinfo "$SERVICE" | head -4
    echo "DNS: $(networksetup -getdnsservers "$SERVICE")"
    ;;
  *)
    echo "사용법: $0 {on [IP]|off|status}" >&2
    exit 1
    ;;
esac
