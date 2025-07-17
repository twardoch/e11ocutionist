"""Tests for version handling and semantic versioning."""

import pytest
from e11ocutionist import __version__


def test_version_exists():
    """Test that version is properly defined."""
    assert __version__ is not None
    assert __version__ != "unknown"
    assert isinstance(__version__, str)


def test_version_format():
    """Test that version follows semantic versioning format."""
    # Allow for development versions (with .dev suffix)
    version_parts = __version__.split('.')
    
    # Should have at least major.minor.patch
    assert len(version_parts) >= 3
    
    # Major version should be numeric
    assert version_parts[0].isdigit()
    
    # Minor version should be numeric
    assert version_parts[1].isdigit()
    
    # Patch version should be numeric (or contain .dev)
    patch_part = version_parts[2]
    if '.dev' in patch_part:
        patch_number = patch_part.split('.dev')[0]
        assert patch_number.isdigit()
    else:
        assert patch_part.isdigit()


def test_version_consistency():
    """Test that version is consistent across imports."""
    from e11ocutionist import __version__ as version1
    from e11ocutionist.__version__ import __version__ as version2
    
    assert version1 == version2