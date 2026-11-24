"""Auto-discover local pentest tools and map them to a user objective.

The agent should never assume only the tools it already knows about. This module
builds a live inventory from PATH and ~/.local/bin, then scores each binary
against the user's task so an objective like "password and encryption auditing
with aircrack-ng" automatically produces aircrack-ng, hashcat, john, and the
matching Kala/GUI routes.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# capability catalog - deliberately broad; discovery still verifies existence.
TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    "aircrack-ng": {"category": "password & encryption auditing", "tags": ["password", "encryption", "wireless", "wpa", "wep", "crack", "capture"], "gui": False, "aliases": ["aircrack"]},
    "airmon-ng": {"category": "wireless auditing", "tags": ["wireless", "monitor", "capture", "wifi"], "gui": False, "aliases": []},
    "airodump-ng": {"category": "wireless auditing", "tags": ["wireless", "capture", "wifi", "recon"], "gui": False, "aliases": []},
    "aireplay-ng": {"category": "wireless auditing", "tags": ["wireless", "wifi", "inject", "deauth"], "gui": False, "aliases": []},
    "wifite": {"category": "wireless auditing", "tags": ["wireless", "wifi", "password", "audit"], "gui": False, "aliases": []},
    "hashcat": {"category": "password & encryption auditing", "tags": ["password", "hash", "encryption", "crack", "gpu", "bruteforce"], "gui": False, "aliases": []},
    "john": {"category": "password & encryption auditing", "tags": ["password", "hash", "encryption", "crack", "cpu", "bruteforce"], "gui": False, "aliases": ["johntheripper", "john the ripper"]},
    "hash-identifier": {"category": "password & encryption auditing", "tags": ["password", "hash", "encryption", "identify", "crypto"], "gui": False, "aliases": ["hashid"]},
    "openssl": {"category": "encryption auditing", "tags": ["encryption", "crypto", "tls", "certificate", "cipher"], "gui": False, "aliases": []},
    "hydra": {"category": "password auditing", "tags": ["password", "login", "bruteforce", "auth", "crack"], "gui": False, "aliases": ["thc-hydra"]},
    "medusa": {"category": "password auditing", "tags": ["password", "login", "bruteforce", "auth", "crack"], "gui": False, "aliases": []},
    "crackmapexec": {"category": "active directory auditing", "tags": ["password", "auth", "active directory", "smb", "crack"], "gui": False, "aliases": ["cme"]},
    "nmap": {"category": "network recon", "tags": ["recon", "ports", "services", "network", "scan"], "gui": False, "aliases": []},
    "zenmap": {"category": "network recon", "tags": ["recon", "ports", "services", "network", "gui", "scan"], "gui": True, "aliases": []},
    "masscan": {"category": "network recon", "tags": ["recon", "ports", "network", "fast", "scan"], "gui": False, "aliases": []},
    "subfinder": {"category": "recon", "tags": ["recon", "subdomains", "dns", "osint"], "gui": False, "aliases": []},
    "amass": {"category": "recon", "tags": ["recon", "subdomains", "dns", "osint"], "gui": False, "aliases": []},
    "theharvester": {"category": "recon", "tags": ["recon", "emails", "subdomains", "osint"], "gui": False, "aliases": ["theHarvester"]},
    "ffuf": {"category": "web auditing", "tags": ["web", "fuzz", "content", "directory", "param"], "gui": False, "aliases": []},
    "gobuster": {"category": "web auditing", "tags": ["web", "fuzz", "content", "directory", "dns"], "gui": False, "aliases": []},
    "dirb": {"category": "web auditing", "tags": ["web", "content", "directory", "bruteforce"], "gui": False, "aliases": []},
    "wfuzz": {"category": "web auditing", "tags": ["web", "fuzz", "param", "injection"], "gui": False, "aliases": []},
    "sqlmap": {"category": "database auditing", "tags": ["web", "sqli", "database", "injection", "sql"], "gui": False, "aliases": []},
    "commix": {"category": "web auditing", "tags": ["web", "command injection", "rce", "injection"], "gui": False, "aliases": []},
    "nikto": {"category": "web auditing", "tags": ["web", "scan", "misconfig"], "gui": False, "aliases": []},
    "curl": {"category": "http client", "tags": ["http", "web", "request", "fetch", "api"], "gui": False, "aliases": []},
    "wget": {"category": "http client", "tags": ["http", "web", "download", "fetch"], "gui": False, "aliases": []},
    "burpsuite": {"category": "web proxy / gui", "tags": ["web", "proxy", "intercept", "repeater", "intruder", "gui"], "gui": True, "aliases": ["burpsuite-atk", "burp"]},
    "wireshark": {"category": "packet analysis", "tags": ["network", "capture", "packets", "protocol", "gui"], "gui": True, "aliases": ["tshark"]},
    "tcpdump": {"category": "packet analysis", "tags": ["network", "capture", "packets", "cli"], "gui": False, "aliases": []},
    "jsql-injection": {"category": "database auditing", "tags": ["web", "sqli", "database", "gui", "injection"], "gui": True, "aliases": ["jsql"]},
    "recon-ng": {"category": "recon", "tags": ["recon", "osint", "web", "gui"], "gui": True, "aliases": []},
    "strings": {"category": "reverse engineering", "tags": ["reverse", "binary", "strings", "static", "firmware"], "gui": False, "aliases": []},
    "volatility": {"category": "forensics", "tags": ["forensics", "memory", "ram", "incident"], "gui": False, "aliases": ["volatility3"]},
    "bloodhound": {"category": "active directory auditing", "tags": ["active directory", "graph", "gui", "ad"], "gui": True, "aliases": []},
    "bettercap": {"category": "network interception", "tags": ["network", "mitm", "capture", "wifi"], "gui": False, "aliases": []},
    "responder": {"category": "network interception", "tags": ["network", "auth", "llmnr", "poison"], "gui": False, "aliases": []},

    # ---- Firmware / embedded reversing ---------------------------------------
    "unsquashfs": {"category": "firmware extraction & analysis", "tags": ["firmware", "squashfs", "filesystem", "extract", "rootfs", "embedded", "reverse"], "gui": False, "aliases": []},
    "mksquashfs": {"category": "firmware extraction & analysis", "tags": ["firmware", "squashfs", "filesystem", "rebuild", "embedded"], "gui": False, "aliases": []},
    "cpio": {"category": "firmware extraction & analysis", "tags": ["firmware", "archive", "extract", "initramfs", "rootfs"], "gui": False, "aliases": []},
    "7z": {"category": "firmware extraction & analysis", "tags": ["firmware", "archive", "extract", "compress", "reverse"], "gui": False, "aliases": ["7za", "p7zip", "7zr"]},
    "unzip": {"category": "firmware extraction & analysis", "tags": ["firmware", "archive", "extract", "zip"], "gui": False, "aliases": []},
    "xz": {"category": "firmware extraction & analysis", "tags": ["firmware", "compress", "lzma", "extract"], "gui": False, "aliases": ["unxz", "xzcat"]},
    "zstd": {"category": "firmware extraction & analysis", "tags": ["firmware", "compress", "extract", "zstandard"], "gui": False, "aliases": ["unzstd"]},
    "xxd": {"category": "binary inspection", "tags": ["firmware", "hex", "dump", "binary", "reverse", "embedded"], "gui": False, "aliases": []},
    "hexdump": {"category": "binary inspection", "tags": ["firmware", "hex", "dump", "binary", "reverse"], "gui": False, "aliases": ["hd"]},
    "od": {"category": "binary inspection", "tags": ["firmware", "hex", "octal", "dump", "binary"], "gui": False, "aliases": []},
    "file": {"category": "binary inspection", "tags": ["firmware", "identify", "magic", "binary", "reverse"], "gui": False, "aliases": []},
    "dd": {"category": "firmware imaging", "tags": ["firmware", "dump", "image", "raw", "flash", "hardware"], "gui": False, "aliases": ["dcfldd"]},
    "fmk": {"category": "firmware extraction & analysis", "tags": ["firmware", "firmware-mod-kit", "extract", "rebuild", "embedded", "router"], "gui": False, "aliases": ["firmware-mod-kit"]},
    "binwalk": {"category": "firmware extraction & analysis", "tags": ["reverse", "firmware", "binary", "extract", "embedded", "iot", "router", "squashfs", "filesystem"], "gui": False, "aliases": []},
    "readelf": {"category": "firmware binary analysis", "tags": ["firmware", "elf", "binary", "reverse", "headers", "embedded"], "gui": False, "aliases": ["llvm-readobj"]},
    "objdump": {"category": "firmware binary analysis", "tags": ["firmware", "disassembly", "binary", "reverse", "elf", "embedded"], "gui": False, "aliases": ["llvm-objdump"]},
    "nm": {"category": "firmware binary analysis", "tags": ["firmware", "symbols", "binary", "reverse", "embedded"], "gui": False, "aliases": []},
    "objcopy": {"category": "firmware binary analysis", "tags": ["firmware", "binary", "reverse", "strip", "embedded"], "gui": False, "aliases": []},
    "gdb": {"category": "firmware debugging", "tags": ["firmware", "debug", "embedded", "reverse", "jtag", "hardware"], "gui": False, "aliases": ["gdb-multiarch"]},
    "radare2": {"category": "firmware binary analysis", "tags": ["firmware", "reverse", "disassembly", "decompile", "binary", "embedded"], "gui": False, "aliases": ["r2", "rabin2"]},
    "rizin": {"category": "firmware binary analysis", "tags": ["firmware", "reverse", "disassembly", "decompile", "binary", "embedded"], "gui": False, "aliases": ["rz-bin"]},
    "ghidra": {"category": "firmware binary analysis", "tags": ["reverse", "firmware", "binary", "decompile", "gui", "embedded", "malware"], "gui": True, "aliases": ["ghidraRun"]},
    "qemu-system-arm": {"category": "firmware emulation", "tags": ["firmware", "qemu", "emulate", "arm", "embedded", "rootfs"], "gui": False, "aliases": []},
    "qemu-system-aarch64": {"category": "firmware emulation", "tags": ["firmware", "qemu", "emulate", "aarch64", "arm", "embedded"], "gui": False, "aliases": []},
    "qemu-system-mips": {"category": "firmware emulation", "tags": ["firmware", "qemu", "emulate", "mips", "router", "embedded"], "gui": False, "aliases": []},
    "qemu-system-mipsel": {"category": "firmware emulation", "tags": ["firmware", "qemu", "emulate", "mips", "mipsel", "router", "embedded"], "gui": False, "aliases": []},
    "qemu-arm": {"category": "firmware emulation", "tags": ["firmware", "qemu", "emulate", "arm", "user-mode", "embedded"], "gui": False, "aliases": []},
    "qemu-mips": {"category": "firmware emulation", "tags": ["firmware", "qemu", "emulate", "mips", "user-mode", "router"], "gui": False, "aliases": []},
    "qemu-mipsel": {"category": "firmware emulation", "tags": ["firmware", "qemu", "emulate", "mipsel", "user-mode", "router"], "gui": False, "aliases": []},
    "chroot": {"category": "firmware emulation", "tags": ["firmware", "rootfs", "chroot", "emulate", "embedded"], "gui": False, "aliases": []},
    "proot": {"category": "firmware emulation", "tags": ["firmware", "rootfs", "emulate", "embedded"], "gui": False, "aliases": []},
    "losetup": {"category": "firmware emulation", "tags": ["firmware", "loop", "image", "mount", "embedded"], "gui": False, "aliases": []},
    "debugfs": {"category": "firmware emulation", "tags": ["firmware", "filesystem", "ext", "image", "embedded"], "gui": False, "aliases": []},

    # ---- Hardware / bus / device reversing -----------------------------------
    "lspci": {"category": "hardware bus & device analysis", "tags": ["hardware", "pci", "device", "enumerate", "reverse"], "gui": False, "aliases": []},
    "lsusb": {"category": "hardware bus & device analysis", "tags": ["hardware", "usb", "device", "enumerate", "reverse"], "gui": False, "aliases": []},
    "lsblk": {"category": "hardware bus & device analysis", "tags": ["hardware", "block", "device", "storage", "enumerate"], "gui": False, "aliases": []},
    "lshw": {"category": "hardware bus & device analysis", "tags": ["hardware", "device", "enumerate", "inventory", "reverse"], "gui": False, "aliases": []},
    "dmidecode": {"category": "hardware bus & device analysis", "tags": ["hardware", "dmi", "bios", "firmware", "enumerate"], "gui": False, "aliases": []},
    "i2cdetect": {"category": "hardware bus & device analysis", "tags": ["hardware", "i2c", "bus", "device", "enumerate", "reverse"], "gui": False, "aliases": []},
    "i2cdump": {"category": "hardware bus & device analysis", "tags": ["hardware", "i2c", "bus", "dump", "device", "reverse"], "gui": False, "aliases": []},
    "i2cget": {"category": "hardware bus & device analysis", "tags": ["hardware", "i2c", "bus", "read", "device"], "gui": False, "aliases": []},
    "i2cset": {"category": "hardware bus & device analysis", "tags": ["hardware", "i2c", "bus", "write", "device"], "gui": False, "aliases": []},
    "flashrom": {"category": "hardware flash tooling", "tags": ["hardware", "firmware", "flash", "spi", "dump", "rom", "reverse"], "gui": False, "aliases": []},
    "openocd": {"category": "hardware debugging", "tags": ["hardware", "jtag", "swd", "debug", "flash", "embedded", "reverse"], "gui": False, "aliases": []},
    "urjtag": {"category": "hardware debugging", "tags": ["hardware", "jtag", "device", "reverse", "embedded"], "gui": False, "aliases": []},
    "avrdude": {"category": "hardware flash tooling", "tags": ["hardware", "avr", "flash", "program", "embedded", "reverse"], "gui": False, "aliases": []},
    "esptool": {"category": "hardware flash tooling", "tags": ["hardware", "esp32", "esp8266", "flash", "firmware", "dump", "embedded"], "gui": False, "aliases": ["esptool.py"]},
    "picocom": {"category": "hardware serial tooling", "tags": ["hardware", "serial", "uart", "console", "device", "reverse"], "gui": False, "aliases": ["minicom", "screen"]},
    "socat": {"category": "hardware serial tooling", "tags": ["hardware", "serial", "socket", "relay", "device"], "gui": False, "aliases": []},
    "stty": {"category": "hardware serial tooling", "tags": ["hardware", "serial", "uart", "console", "device"], "gui": False, "aliases": []},
    "sigrok-cli": {"category": "hardware signal analysis", "tags": ["hardware", "logic", "analyzer", "signal", "spi", "i2c", "reverse"], "gui": False, "aliases": []},
    "pulseview": {"category": "hardware signal analysis", "tags": ["hardware", "logic", "analyzer", "signal", "gui", "reverse"], "gui": True, "aliases": []},
    "rtl_fm": {"category": "hardware SDR tooling", "tags": ["hardware", "sdr", "radio", "rtl-sdr", "signal", "reverse"], "gui": False, "aliases": ["rtl_sdr"]},
    "hackrf_info": {"category": "hardware SDR tooling", "tags": ["hardware", "sdr", "radio", "hackrf", "signal", "reverse"], "gui": False, "aliases": []},
    "dfu-util": {"category": "hardware flash tooling", "tags": ["hardware", "usb", "dfu", "flash", "firmware", "embedded"], "gui": False, "aliases": []},
    "smartctl": {"category": "hardware storage analysis", "tags": ["hardware", "storage", "nvme", "disk", "firmware", "device"], "gui": False, "aliases": []},
    "nvme": {"category": "hardware storage analysis", "tags": ["hardware", "nvme", "storage", "firmware", "device"], "gui": False, "aliases": []},
    "hdparm": {"category": "hardware storage analysis", "tags": ["hardware", "storage", "ata", "device", "firmware"], "gui": False, "aliases": []},
    "arduino-cli": {"category": "hardware embedded tooling", "tags": ["hardware", "arduino", "embedded", "mcu", "program"], "gui": False, "aliases": ["platformio"]},
}

_WORD_RE = re.compile(r"[a-z0-9+_.-]+")


def _norm(text: Any) -> str:
    return str(text or "").lower().strip()


_INFER_RULES = [
    ("squashfs", "firmware extraction & analysis", ["firmware", "squashfs", "filesystem", "extract", "rootfs"], False),
    ("binwalk", "firmware extraction & analysis", ["firmware", "extract", "embedded", "reverse"], False),
    ("firmware", "firmware extraction & analysis", ["firmware", "embedded", "reverse"], False),
    ("objdump", "firmware binary analysis", ["firmware", "disassembly", "binary", "reverse", "embedded"], False),
    ("readelf", "firmware binary analysis", ["firmware", "elf", "headers", "binary", "embedded"], False),
    ("readobj", "firmware binary analysis", ["firmware", "elf", "headers", "binary", "embedded"], False),
    ("gdb", "firmware debugging", ["firmware", "debug", "jtag", "embedded", "reverse"], False),
    ("qemu", "firmware emulation", ["firmware", "qemu", "emulate", "rootfs", "embedded"], False),
    ("openocd", "hardware debugging", ["hardware", "jtag", "swd", "embedded", "reverse"], False),
    ("jtag", "hardware debugging", ["hardware", "jtag", "embedded", "reverse"], False),
    ("esptool", "hardware flash tooling", ["hardware", "esp32", "esp8266", "flash", "firmware"], False),
    ("flashrom", "hardware flash tooling", ["hardware", "flash", "spi", "rom", "firmware"], False),
    ("avrdude", "hardware flash tooling", ["hardware", "avr", "mcu", "flash", "embedded"], False),
    ("dfu", "hardware flash tooling", ["hardware", "usb", "dfu", "flash", "firmware"], False),
    ("i2c", "hardware bus & device analysis", ["hardware", "i2c", "bus", "device", "reverse"], False),
    ("spi", "hardware bus & device analysis", ["hardware", "spi", "bus", "device", "reverse"], False),
    ("i2s", "hardware bus & device analysis", ["hardware", "i2s", "bus", "device"], False),
    ("serial", "hardware serial tooling", ["hardware", "serial", "uart", "console", "device"], False),
    ("uart", "hardware serial tooling", ["hardware", "serial", "uart", "console", "device"], False),
    ("logic", "hardware signal analysis", ["hardware", "logic", "analyzer", "signal", "reverse"], False),
    ("sigrok", "hardware signal analysis", ["hardware", "logic", "analyzer", "signal"], False),
    ("rtl", "hardware SDR tooling", ["hardware", "sdr", "radio", "signal"], False),
    ("hackrf", "hardware SDR tooling", ["hardware", "sdr", "radio", "signal"], False),
    ("mcu", "hardware embedded tooling", ["hardware", "mcu", "embedded", "program"], False),
    ("arm-none-eabi", "hardware embedded tooling", ["hardware", "arm", "embedded", "cross-compile"], False),
    ("aarch64-linux", "hardware embedded tooling", ["hardware", "aarch64", "embedded", "cross-compile"], False),
    ("mips-linux", "hardware embedded tooling", ["hardware", "mips", "router", "cross-compile"], False),
    ("riscv", "hardware embedded tooling", ["hardware", "riscv", "embedded", "cross-compile"], False),
]


def _infer_entry(name: str) -> Optional[Dict[str, Any]]:
    """Classify cross-tool binaries such as arm-none-eabi-objdump."""
    needle = _norm(name)
    for marker, category, tags, gui in _INFER_RULES:
        if marker in needle:
            return {
                "name": name,
                "category": category,
                "tags": list(tags),
                "gui": gui,
                "aliases": [],
                "inferred": True,
            }
    return None


def _catalog_entry(name: str) -> Dict[str, Any]:
    canonical = _norm(name)
    for key, meta in TOOL_CATALOG.items():
        if canonical == key or canonical in [_norm(a) for a in meta.get("aliases", [])]:
            entry = dict(meta)
            entry["name"] = key
            return entry
    inferred = _infer_entry(name)
    if inferred is not None:
        return inferred
    return {
        "name": name,
        "category": "general",
        "tags": [],
        "gui": False,
        "aliases": [],
    }


def discover_tools(extra_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Return verified local tools with capability tags and resolved paths."""
    names = set(TOOL_CATALOG)
    for item in extra_names or []:
        if str(item).strip():
            names.add(str(item).strip())
    found: Dict[str, Dict[str, Any]] = {}
    local_bin = Path.home() / ".local" / "bin"
    for name in sorted(names):
        path = shutil.which(name)
        if not path and local_bin.is_dir():
            candidate = local_bin / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                path = str(candidate)
        if not path:
            continue
        entry = _catalog_entry(name)
        entry.update({"path": path, "kind": "binary", "available": True})
        found.setdefault(entry["name"], entry)
    return sorted(found.values(), key=lambda x: x["name"].lower())


def mcp_names_for_tools(tools: Sequence[Dict[str, Any]], mcp_tool_names: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Map local tool names to matching Kala/MCP tool names when present."""
    mapping: Dict[str, str] = {}
    for tool in tools:
        needle = re.sub(r"[^a-z0-9]", "", _norm(tool["name"]))
        for mcp_name in mcp_tool_names or []:
            compact = re.sub(r"[^a-z0-9]", "", _norm(mcp_name))
            if needle and needle in compact:
                mapping[tool["name"]] = str(mcp_name)
                break
    return mapping


def rank_tools(
    task: str,
    tools: Sequence[Dict[str, Any]],
    *,
    mcp_tool_names: Optional[Iterable[str]] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Score discovered tools against the objective and explain each match."""
    text = _norm(task)
    words = set(_WORD_RE.findall(text))
    mcp_map = mcp_names_for_tools(tools, mcp_tool_names)
    ranked: List[Dict[str, Any]] = []
    for tool in tools:
        score = 0
        reasons: List[str] = []
        names = [tool["name"], *tool.get("aliases", [])]
        for name in names:
            needle = _norm(name)
            if needle and needle in text:
                score += 120
                reasons.append(f"explicit tool name: {name}")
        for tag in tool.get("tags", []):
            tag_norm = _norm(tag)
            if tag_norm and tag_norm in text:
                score += 14
                reasons.append(f"tag {tag}")
            elif tag_norm and tag_norm in words:
                score += 10
        category = _norm(tool.get("category"))
        if category and category in text:
            score += 18
            reasons.append(f"category {tool['category']}")
        else:
            for word in _WORD_RE.findall(category):
                if word in words:
                    score += 6
                    reasons.append(f"category word {word}")
        mcp = mcp_map.get(tool["name"])
        if mcp:
            score += 35
            reasons.append(f"Kala/MCP wrapper: {mcp}")
        if "gui" in words and tool.get("gui"):
            score += 20
            reasons.append("GUI route available")
        if score <= 0:
            continue
        ranked.append({
            **tool,
            "score": score,
            "reason": "; ".join(dict.fromkeys(reasons))[:300],
            "mcp_tool": mcp,
        })
    ranked.sort(key=lambda x: (-x["score"], x["name"].lower()))
    return ranked[: max(0, limit)]


def build_routes(task: str, ranked: Sequence[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    """Produce bounded, ordered strategies for the top matching tools."""
    routes: List[Dict[str, Any]] = []
    for tool in ranked[: max(0, limit)]:
        name = tool["name"]
        if tool.get("mcp_tool"):
            routes.append({
                "route": "kala",
                "tool": name,
                "mcp_tool": tool["mcp_tool"],
                "priority": 1,
                "next": f"call {tool['mcp_tool']} with task-derived options",
            })
        routes.append({
            "route": "cli",
            "tool": name,
            "priority": 2,
            "next": f"run {name} --help, then build a minimal command for: {task[:120]}",
            "probe": f"{name} --help",
        })
        if tool.get("gui"):
            app = "burpsuite-atk" if name == "burpsuite" else name
            routes.append({
                "route": "gui",
                "tool": name,
                "priority": 3,
                "next": f"desktop_launch({app}) -> desktop_wait -> desktop_snapshot -> click/type -> verify",
            })
        routes.append({
            "route": "script",
            "tool": name,
            "priority": 4,
            "next": "write a bounded helper script that parses output and records evidence",
        })
    routes.sort(key=lambda x: x["priority"])
    return routes


def inventory_summary(tools: Sequence[Dict[str, Any]]) -> str:
    if not tools:
        return "no matching local tools"
    return ", ".join(f"{t['name']} ({t['category']})" for t in tools[:12])
