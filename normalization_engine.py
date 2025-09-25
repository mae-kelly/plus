import re
import unicodedata
from typing import Any, Union, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class NormalizationEngine:
    
    def __init__(self):
        self.hostname_rules = {
            'lowercase': True,
            'trim_whitespace': True,
            'remove_quotes': True,
            'handle_unicode': True,
            'normalize_separators': True
        }
        
        self.variations = {
            'whitespace': [' ', '\t', '\n', '\r', '\xa0', '\u00a0', '\u2000', '\u2001', 
                          '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007',
                          '\u2008', '\u2009', '\u200a', '\u202f', '\u205f', '\u3000'],
            'quotes': ['"', "'", '`', '"', '"', ''', ''', '‚', '„', '‹', '›', 
                      '«', '»', '｢', '｣'],
            'dashes': ['-', '–', '—', '­', '‐', '‑', '‒', '―', '−'],
            'dots': ['.', '。', '․', '‧', '⁘', '⁙', '⁚', '⁛'],
            'underscores': ['_', '＿', '‿', '⁀']
        }
        
        self.cache = {}
        
    def normalize(self, value: Any) -> str:
        if value is None:
            return ''
            
        str_val = str(value)
        
        if str_val in self.cache:
            return self.cache[str_val]
        
        normalized = self._normalize_string(str_val)
        self.cache[str_val] = normalized
        
        return normalized
        
    def normalize_hostname(self, hostname: str) -> str:
        if not hostname:
            return ''
            
        hostname = str(hostname).strip()
        
        # Remove all types of whitespace
        for ws in self.variations['whitespace']:
            hostname = hostname.replace(ws, '')
        
        # Remove wrapping quotes
        hostname = self._remove_wrapper_quotes(hostname)
        
        # Normalize unicode
        hostname = self._normalize_unicode(hostname)
        
        # Lowercase everything
        hostname = hostname.lower()
        
        # Normalize separators
        hostname = self._normalize_separators(hostname)
        
        # Remove multiple consecutive dots
        while '..' in hostname:
            hostname = hostname.replace('..', '.')
            
        # Remove trailing/leading dots
        hostname = hostname.strip('.')
        
        # Handle special cases
        hostname = self._handle_special_cases(hostname)
        
        return hostname
        
    def normalize_ip(self, ip: str) -> str:
        if not ip:
            return ''
            
        ip = str(ip).strip()
        
        # Remove all whitespace
        for ws in self.variations['whitespace']:
            ip = ip.replace(ws, '')
            
        # Check if IPv6
        if ':' in ip:
            return self._normalize_ipv6(ip)
        else:
            return self._normalize_ipv4(ip)
            
    def normalize_column_value(self, value: Any, column_name: str = None) -> str:
        if value is None:
            return ''
            
        str_val = str(value).strip()
        
        if column_name:
            col_lower = column_name.lower()
            
            # Hostname columns
            if any(term in col_lower for term in ['host', 'server', 'node', 'machine', 'instance']):
                return self.normalize_hostname(str_val)
                
            # IP columns
            elif any(term in col_lower for term in ['ip', 'address', 'ipv4', 'ipv6']):
                return self.normalize_ip(str_val)
                
            # Domain columns
            elif 'domain' in col_lower or 'dns' in col_lower or 'fqdn' in col_lower:
                return self.normalize_hostname(str_val)
                
            # Email columns
            elif 'email' in col_lower or 'mail' in col_lower:
                return self._normalize_email(str_val)
                
        return self.normalize(str_val)
        
    def _normalize_string(self, text: str) -> str:
        if not text:
            return ''
            
        text = text.strip()
        
        # Normalize whitespace
        for ws in self.variations['whitespace']:
            text = text.replace(ws, ' ')
            
        # Collapse multiple spaces
        text = ' '.join(text.split())
        
        # Handle unicode
        text = self._normalize_unicode(text)
        
        return text
        
    def _normalize_unicode(self, text: str) -> str:
        try:
            # Normalize unicode (NFKD - compatibility decomposition)
            text = unicodedata.normalize('NFKD', text)
            
            # Remove non-ASCII characters if they're not essential
            allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_@/:')
            text = ''.join(c for c in text if c in allowed or ord(c) < 128)
            
        except Exception as e:
            logger.debug(f"Unicode normalization error: {e}")
            
        return text
        
    def _remove_wrapper_quotes(self, text: str) -> str:
        if len(text) < 2:
            return text
            
        for quote in self.variations['quotes']:
            if text.startswith(quote) and text.endswith(quote):
                return text[1:-1]
                
        return text
        
    def _normalize_separators(self, text: str) -> str:
        # Normalize dashes to standard hyphen
        for dash in self.variations['dashes']:
            if dash != '-':
                text = text.replace(dash, '-')
                
        # Normalize dots to standard period
        for dot in self.variations['dots']:
            if dot != '.':
                text = text.replace(dot, '.')
                
        # Normalize underscores
        for underscore in self.variations['underscores']:
            if underscore != '_':
                text = text.replace(underscore, '_')
                
        return text
        
    def _normalize_ipv4(self, ip: str) -> str:
        try:
            parts = ip.split('.')
            
            if len(parts) != 4:
                return ip
                
            normalized_parts = []
            for part in parts:
                try:
                    num = int(part)
                    if 0 <= num <= 255:
                        normalized_parts.append(str(num))
                    else:
                        return ip
                except ValueError:
                    return ip
                    
            return '.'.join(normalized_parts)
            
        except Exception:
            return ip
            
    def _normalize_ipv6(self, ip: str) -> str:
        try:
            parts = ip.split(':')
            normalized_parts = []
            
            for part in parts:
                if part == '':
                    normalized_parts.append('')
                else:
                    normalized = part.lstrip('0') or '0'
                    normalized_parts.append(normalized.lower())
                    
            return ':'.join(normalized_parts)
            
        except Exception:
            return ip
            
    def _normalize_email(self, email: str) -> str:
        if '@' not in email:
            return email
            
        try:
            local, domain = email.rsplit('@', 1)
            domain = self.normalize_hostname(domain)
            local = local.strip().lower()
            return f"{local}@{domain}"
            
        except Exception:
            return email.lower()
            
    def _handle_special_cases(self, hostname: str) -> str:
        # Remove common prefixes that aren't part of actual hostname
        prefixes_to_remove = ['http://', 'https://', 'ftp://', 'ssh://', 'telnet://']
        for prefix in prefixes_to_remove:
            if hostname.startswith(prefix):
                hostname = hostname[len(prefix):]
                
        # Remove port numbers
        if ':' in hostname and not self._looks_like_ipv6(hostname):
            hostname = hostname.split(':')[0]
            
        # Remove URL paths
        if '/' in hostname:
            hostname = hostname.split('/')[0]
            
        # Remove username@
        if '@' in hostname:
            parts = hostname.split('@')
            if len(parts) == 2:
                hostname = parts[1]
                
        return hostname
        
    def _looks_like_ipv6(self, text: str) -> bool:
        if text.count(':') >= 2:
            hex_chars = set('0123456789abcdefABCDEF:')
            return all(c in hex_chars for c in text.replace('.', ''))
            
        return False
        
    def get_stats(self) -> dict:
        return {
            'cache_size': len(self.cache),
            'cache_memory_bytes': sum(len(k) + len(v) for k, v in self.cache.items())
        }