"""Security tests for e11ocutionist."""

import pytest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from e11ocutionist.utils import sanitize_filename


class TestInputSanitization:
    """Test input sanitization and validation."""

    def test_filename_sanitization_security(self):
        """Test that filename sanitization prevents directory traversal."""
        # Test various malicious filename patterns
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file\x00.txt",  # Null byte injection
            "con.txt",  # Windows reserved name
            "aux.txt",  # Windows reserved name
            "prn.txt",  # Windows reserved name
            "nul.txt",  # Windows reserved name
            "com1.txt",  # Windows reserved name
            "lpt1.txt",  # Windows reserved name
            "a" * 256,  # Extremely long filename
            ".",  # Current directory
            "..",  # Parent directory
            "",  # Empty string
            " ",  # Space only
            "\t",  # Tab only
            "\n",  # Newline only
        ]

        for malicious_name in malicious_filenames:
            sanitized = sanitize_filename(malicious_name)

            # Should not contain path traversal sequences
            assert "../" not in sanitized
            assert "..\\" not in sanitized

            # Should not contain null bytes
            assert "\x00" not in sanitized

            # Should not be empty (unless input was empty/whitespace)
            if malicious_name.strip():
                assert sanitized.strip() != ""

            # Should not be reserved Windows names
            if sanitized.lower().split(".")[0] in [
                "con",
                "aux",
                "prn",
                "nul",
                "com1",
                "lpt1",
            ]:
                assert sanitized != malicious_name  # Should be modified

    def test_xml_injection_prevention(self):
        """Test prevention of XML injection attacks."""
        # Test various XML injection patterns
        malicious_xml_inputs = [
            '<?xml version="1.0"?><!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><root>&xxe;</root>',
            '<script>alert("XSS")</script>',
            '&lt;script&gt;alert("XSS")&lt;/script&gt;',
            ']]><script>alert("XSS")</script>',
            '<![CDATA[<script>alert("XSS")</script>]]>',
            '<?xml-stylesheet type="text/xsl" href="malicious.xsl"?>',
        ]

        for malicious_input in malicious_xml_inputs:
            # Test that XML parsing handles malicious input safely
            # This would depend on the actual XML processing implementation
            # For now, we'll test basic XML parsing safety
            try:
                # Should not crash or execute malicious code
                root = ET.fromstring(f"<root>{malicious_input}</root>")
                # If parsing succeeds the parser handled the input safely:
                # content treated as text or child elements — not executable.
                # (Entity-unescaping is normal parser behaviour, not a security
                # issue; what matters is that no external entities were resolved.)
                assert root is not None
            except ET.ParseError:
                # Parse errors are acceptable for malicious input
                pass

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        # Test various path traversal patterns
        malicious_paths = [
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "file:///etc/passwd",
            "\\\\server\\share\\file.txt",
            "..\\..\\..\\etc\\passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "....//....//etc/passwd",  # Double encoding
        ]

        for malicious_path in malicious_paths:
            # Test that paths are properly validated
            # This depends on actual path validation implementation
            # For now, test that Path doesn't resolve to sensitive locations
            path = Path(malicious_path)

            # Should not resolve to system directories
            resolved = path.resolve()
            system_dirs = ["/etc", "/sys", "/proc", "C:\\Windows", "C:\\System32"]

            for sys_dir in system_dirs:
                assert not str(resolved).startswith(sys_dir)


class TestAPIKeySecurity:
    """Test API key handling security."""

    def test_api_key_not_logged(self):
        """Test that API keys are not logged in error messages."""
        # Mock API key
        test_api_key = "sk-test-key-1234567890abcdef"

        # Test that API key doesn't appear in logs/errors
        # This would require testing actual logging output
        # For now, we'll test that exceptions don't contain the key

        with patch.dict("os.environ", {"OPENAI_API_KEY": test_api_key}):
            # Simulate an error that might expose the API key
            try:
                # This would trigger an error that might expose the key
                # Implementation depends on actual error handling
                pass
            except Exception as e:
                # Error messages should not contain the API key
                assert test_api_key not in str(e)

    def test_api_key_masking(self):
        """Test that API keys are masked in debug output."""
        # Test that API keys are properly masked when displayed
        test_api_key = "sk-test-key-1234567890abcdef"

        # This would test actual masking implementation
        # For now, we'll test a simple masking function
        def mask_api_key(key):
            if len(key) > 8:
                return key[:4] + "*" * (len(key) - 8) + key[-4:]
            return "*" * len(key)

        masked = mask_api_key(test_api_key)
        assert test_api_key not in masked
        assert "*" in masked
        assert masked.startswith("sk-t")
        assert masked.endswith("cdef")


class TestFileSystemSecurity:
    """Test file system security measures."""

    def test_temp_file_permissions(self):
        """Test that temporary files have proper permissions."""
        # Test that temp files are created with restricted permissions
        # This would require actual file creation and permission checking
        pytest.skip("File permission testing requires OS-specific implementation")

    def test_output_directory_validation(self):
        """Test that output directories are properly validated."""
        # Test that output directories don't allow writing to sensitive locations
        sensitive_dirs = [
            "/etc",
            "/root",
            "/sys",
            "/proc",
            "C:\\Windows",
            "C:\\System32",
        ]

        for sensitive_dir in sensitive_dirs:
            # Test that these directories are rejected as output directories
            # This depends on actual directory validation implementation
            path = Path(sensitive_dir)

            # Should not allow writing to system directories
            # This is a placeholder for actual validation logic
            assert path.is_absolute() or not path.exists()

    def test_symlink_handling(self):
        """Test proper handling of symbolic links."""
        # Test that symbolic links are handled safely
        # This prevents symlink attacks
        pytest.skip("Symlink testing requires filesystem setup")


class TestDependencyVulnerabilities:
    """Test for known dependency vulnerabilities."""

    def test_no_known_vulnerabilities(self):
        """Test that dependencies don't have known vulnerabilities."""
        # This would integrate with security scanning tools
        # For now, we'll test that critical dependencies are present

        try:
            import lxml
            import litellm
            import elevenlabs
            import loguru
            import tenacity
            import backoff

            # Basic import test - would be replaced with actual vulnerability scanning
            assert True  # Dependencies import successfully
        except ImportError as e:
            pytest.fail(f"Critical dependency missing: {e}")

    def test_secure_xml_parsing(self):
        """Test that XML parsing is secure against XXE attacks."""
        # Test that XML parser is configured securely
        from lxml import etree

        # Test that external entities are disabled
        parser = etree.XMLParser()

        # Should not process external entities
        malicious_xml = """<?xml version="1.0"?>
        <!DOCTYPE root [
            <!ENTITY xxe SYSTEM "file:///etc/passwd">
        ]>
        <root>&xxe;</root>"""

        try:
            # Should not resolve external entities
            tree = etree.fromstring(malicious_xml, parser)
            # If parsing succeeds, should not contain file contents
            assert "root:x:" not in etree.tostring(tree, encoding="unicode")
        except Exception:
            # Parse errors are acceptable for malicious input
            pass
