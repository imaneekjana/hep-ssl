#!/usr/bin/env python3
"""Standard-library integrity and extraction helpers for execution-node jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


def require_basename(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value) or value in {'.', '..'}:
        raise ValueError(f'Expected a plain transfer-file basename, got {value!r}.')
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata_fingerprint(metadata):
    """Exactly matches src.training.checkpoint, without importing torch."""
    content = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_safe_archive(archive, destination, expected_root=None):
    """Extract only regular files/directories after validating EVERY member.

    Symlinks, hardlinks, devices, FIFOs, absolute paths, parent traversal,
    duplicate file destinations and overwriting existing files are unsupported.
    Self-contained raw archives are required; no link is followed or invented.
    """
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError(f"Extraction destination must not be a symlink: {destination}")
    destination = destination.resolve()
    if expected_root is not None:
        expected_root = require_basename(expected_root)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f'Extraction destination is not a directory: {destination}')
    with tarfile.open(archive, 'r:*') as stream:
        checked, seen = [], {}
        for member in stream.getmembers():
            name = member.name
            path = PurePosixPath(name)
            if not name or '\\' in name or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe archive path: {name!r}')
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsupported archive member {name!r}: links and special files are not allowed. Supply a self-contained archive.')
            parts = tuple(part for part in path.parts if part != '.')
            if not parts:
                if not member.isdir() or expected_root is not None:
                    raise ValueError(f'Unexpected root member: {name!r}')
                continue
            if expected_root is not None and parts[0] != expected_root:
                raise ValueError(f'Archive member {name!r} is outside required root {expected_root!r}.')
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Archive path leaves destination: {name!r}')
            for ancestor in (target, *target.parents):
                if ancestor == destination.parent:
                    break
                if ancestor.is_symlink():
                    raise ValueError(f'Extraction would follow an existing symlink: {ancestor}')
            relative = '/'.join(parts)
            if relative in seen:
                if member.isdir() and seen[relative] == 'directory':
                    continue
                raise ValueError(f'Duplicate archive destination: {name!r}')
            kind = 'directory' if member.isdir() else 'file'
            seen[relative] = kind
            if target.exists() and (not member.isdir() or not target.is_dir()):
                raise FileExistsError(f'Extraction will not overwrite {target}')
            checked.append((member, target))
        # A file cannot also be the parent of another member, regardless of order.
        for relative, kind in seen.items():
            for parent in PurePosixPath(relative).parents:
                if str(parent) in seen and seen[str(parent)] != 'directory':
                    raise ValueError(f'Archive file is also a parent directory: {parent}')
        if expected_root is not None and not checked:
            raise ValueError(f'Archive has no members under {expected_root!r}.')
        destination.mkdir(parents=True, exist_ok=True)
        for member, target in checked:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f'Cannot read archive member {member.name!r}')
            with source, target.open('xb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            target.chmod(member.mode & 0o777)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='command', required=True)
    extract = actions.add_parser('extract')
    extract.add_argument('archive')
    extract.add_argument('destination')
    extract.add_argument('--expected-root')
    args = parser.parse_args(argv)
    extract_safe_archive(args.archive, args.destination, args.expected_root)


if __name__ == '__main__':
    main()
