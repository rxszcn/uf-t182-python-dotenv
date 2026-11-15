import io
import logging
import os
import pathlib
import stat
import sys
import tempfile
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import (
    IO,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .parser import Binding, parse_stream
from .variables import Variable, parse_variables

# A type alias for a string path to be used for the paths in this file.
# These paths may flow to `open()` and `os.replace()`.
StrPath = Union[str, "os.PathLike[str]"]

logger = logging.getLogger(__name__)


class CircularInterpolationError(ValueError):
    """Raised when keys in a .env file reference each other in a cycle.

    A self-reference such as ``A=${A}`` is not a cycle: when ``A`` is being
    resolved it is not yet part of the values collected from the file, so the
    reference falls back to the environment, the default, or an empty string.
    Two or more keys that reference each other have no such fallback: the
    value of each depends on the value of the other, so there is nothing to
    fall back on and the keys would otherwise silently become empty strings.

    ``cycles`` holds every detected cycle as a tuple of key names whose first
    and last element are the same key, in file order.
    """

    # Cycles longer than this are abbreviated in the message, which keeps a
    # generated .env with a huge tangled group readable. `cycles` is complete.
    _MAX_KEYS_IN_MESSAGE = 12

    def __init__(self, cycles: Sequence[Sequence[str]]) -> None:
        self.cycles: List[Tuple[str, ...]] = [tuple(cycle) for cycle in cycles]
        description = "; ".join(self._format_cycle(cycle) for cycle in self.cycles)
        super().__init__(f"circular variable reference detected: {description}")

    @classmethod
    def _format_cycle(cls, cycle: Sequence[str]) -> str:
        if len(cycle) <= cls._MAX_KEYS_IN_MESSAGE:
            return " -> ".join(cycle)
        head = " -> ".join(cycle[: cls._MAX_KEYS_IN_MESSAGE - 2])
        return f"{head} -> ... ({len(cycle) - 1} keys) ... -> {cycle[-1]}"


def _load_dotenv_disabled() -> bool:
    """
    Determine if dotenv loading has been disabled.
    """
    if "PYTHON_DOTENV_DISABLED" not in os.environ:
        return False
    value = os.environ["PYTHON_DOTENV_DISABLED"].casefold()
    return value in {"1", "true", "t", "yes", "y"}


def with_warn_for_invalid_lines(mappings: Iterator[Binding]) -> Iterator[Binding]:
    for mapping in mappings:
        if mapping.error:
            logger.warning(
                "python-dotenv could not parse statement starting at line %s",
                mapping.original.line,
            )
        yield mapping


class DotEnv:
    def __init__(
        self,
        dotenv_path: Optional[StrPath],
        stream: Optional[IO[str]] = None,
        verbose: bool = False,
        encoding: Optional[str] = None,
        interpolate: bool = True,
        override: bool = True,
    ) -> None:
        self.dotenv_path: Optional[StrPath] = dotenv_path
        if isinstance(dotenv_path, str):
            self.dotenv_path = os.path.expanduser(dotenv_path)
        elif dotenv_path is not None:
            self.dotenv_path = pathlib.Path(dotenv_path).expanduser()
        self.stream: Optional[IO[str]] = stream
        self._dict: Optional[Dict[str, Optional[str]]] = None
        self.verbose: bool = verbose
        self.encoding: Optional[str] = encoding
        self.interpolate: bool = interpolate
        self.override: bool = override

    @contextmanager
    def _get_stream(self) -> Iterator[IO[str]]:
        if self.dotenv_path and _is_file_or_fifo(self.dotenv_path):
            with open(self.dotenv_path, encoding=self.encoding) as stream:
                yield stream
        elif self.stream is not None:
            yield self.stream
        else:
            if self.verbose:
                logger.info(
                    "python-dotenv could not find configuration file %s.",
                    self.dotenv_path or ".env",
                )
            yield io.StringIO("")

    def dict(self) -> Dict[str, Optional[str]]:
        """Return dotenv as dict"""
        if self._dict is not None:
            return self._dict

        raw_values = self.parse()

        if self.interpolate:
            self._dict = OrderedDict(
                resolve_variables(raw_values, override=self.override)
            )
        else:
            self._dict = OrderedDict(raw_values)

        return self._dict

    def parse(self) -> Iterator[Tuple[str, Optional[str]]]:
        with self._get_stream() as stream:
            for mapping in with_warn_for_invalid_lines(parse_stream(stream)):
                if mapping.key is not None:
                    yield mapping.key, mapping.value

    def set_as_environment_variables(self) -> bool:
        """
        Load the current dotenv as system environment variable.
        """
        if not self.dict():
            return False

        for k, v in self.dict().items():
            if k in os.environ and not self.override:
                continue
            if v is not None:
                os.environ[k] = v

        return True

    def get(self, key: str) -> Optional[str]:
        """ """
        data = self.dict()

        if key in data:
            return data[key]

        if self.verbose:
            logger.warning("Key %s not found in %s.", key, self.dotenv_path)

        return None


def get_key(
    dotenv_path: StrPath,
    key_to_get: str,
    encoding: Optional[str] = "utf-8",
) -> Optional[str]:
    """
    Get the value of a given key from the given .env.

    Returns `None` if the key isn't found or doesn't have a value.
    """
    return DotEnv(dotenv_path, verbose=True, encoding=encoding).get(key_to_get)


def _discard_temp_file(path: pathlib.Path) -> None:
    """
    Delete `rewrite`'s temporary file, ignoring any failure to do so.

    This runs while another exception is propagating, so it must not raise:
    that error is the one worth reporting. On Windows, a file whose mode has
    no owner-write bit carries the read-only attribute and can't be unlinked,
    so the mode is reset before a second attempt.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        try:
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("python-dotenv could not remove the temporary file %s", path)


@contextmanager
def rewrite(
    path: StrPath,
    encoding: Optional[str],
    follow_symlinks: bool = False,
) -> Iterator[Tuple[IO[str], IO[str]]]:
    if follow_symlinks:
        path = os.path.realpath(path)

    try:
        source: IO[str] = open(path, encoding=encoding)
        try:
            path_stat = os.lstat(path)
            original_mode: Optional[int] = (
                stat.S_IMODE(path_stat.st_mode)
                if stat.S_ISREG(path_stat.st_mode)
                else None
            )
        except BaseException:
            source.close()
            raise
    except FileNotFoundError:
        source = io.StringIO("")
        original_mode = None

    try:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            delete=False,
            prefix=".tmp_",
            dir=os.path.dirname(os.path.abspath(path)),
        )
    except OSError as err:
        source.close()
        # Report the target path, not the name of the temporary file.
        err.filename = os.fspath(path)
        raise

    with temp_file as dest:
        dest_path = pathlib.Path(dest.name)
        error = None

        try:
            with source:
                yield (source, dest)
        except BaseException as err:
            error = err

    if error is None:
        try:
            if original_mode is not None:
                os.chmod(dest_path, original_mode)

            os.replace(dest_path, path)
        except BaseException:
            _discard_temp_file(dest_path)
            raise
    else:
        _discard_temp_file(dest_path)
        raise error from None


def set_key(
    dotenv_path: StrPath,
    key_to_set: str,
    value_to_set: str,
    quote_mode: str = "always",
    export: bool = False,
    encoding: Optional[str] = "utf-8",
    follow_symlinks: bool = False,
) -> Tuple[Optional[bool], str, str]:
    """
    Adds or Updates a key/value to the given .env

    The target .env file is created if it doesn't exist.

    This function doesn't follow symlinks by default, to avoid accidentally
    modifying a file at a potentially untrusted path. If you don't need this
    protection and need symlinks to be followed, use `follow_symlinks`.
    """
    if isinstance(dotenv_path, str):
        dotenv_path = os.path.expanduser(dotenv_path)
    else:
        dotenv_path = pathlib.Path(dotenv_path).expanduser()
    if quote_mode not in ("always", "auto", "never"):
        raise ValueError(f"Unknown quote_mode: {quote_mode}")

    quote = quote_mode == "always" or (
        quote_mode == "auto" and not value_to_set.isalnum()
    )

    if quote:
        # The single-quoted-value parser decodes `\\` and `\'`, so both have to
        # be escaped here for the value to survive a write/read round-trip.
        # Backslashes first, otherwise the backslash added by the quote
        # escaping would be escaped in turn.
        escaped = value_to_set.replace("\\", "\\\\").replace("'", "\\'")
        value_out = f"'{escaped}'"
    else:
        value_out = value_to_set
    if export:
        line_out = f"export {key_to_set}={value_out}\n"
    else:
        line_out = f"{key_to_set}={value_out}\n"

    with rewrite(dotenv_path, encoding=encoding, follow_symlinks=follow_symlinks) as (
        source,
        dest,
    ):
        replaced = False
        missing_newline = False
        for mapping in with_warn_for_invalid_lines(parse_stream(source)):
            if mapping.key == key_to_set:
                dest.write(line_out)
                replaced = True
            else:
                dest.write(mapping.original.string)
                missing_newline = not mapping.original.string.endswith("\n")
        if not replaced:
            if missing_newline:
                dest.write("\n")
            dest.write(line_out)

    return True, key_to_set, value_to_set


def unset_key(
    dotenv_path: StrPath,
    key_to_unset: str,
    quote_mode: str = "always",
    encoding: Optional[str] = "utf-8",
    follow_symlinks: bool = False,
) -> Tuple[Optional[bool], str]:
    """
    Removes a given key from the given `.env` file.

    If the .env path given doesn't exist, fails.
    If the given key doesn't exist in the .env, fails.

    This function doesn't follow symlinks by default, to avoid accidentally
    modifying a file at a potentially untrusted path. If you don't need this
    protection and need symlinks to be followed, use `follow_symlinks`.
    """
    if isinstance(dotenv_path, str):
        dotenv_path = os.path.expanduser(dotenv_path)
    else:
        dotenv_path = pathlib.Path(dotenv_path).expanduser()
    if not os.path.exists(dotenv_path):
        logger.warning("Can't delete from %s - it doesn't exist.", dotenv_path)
        return None, key_to_unset

    removed = False
    with rewrite(dotenv_path, encoding=encoding, follow_symlinks=follow_symlinks) as (
        source,
        dest,
    ):
        for mapping in with_warn_for_invalid_lines(parse_stream(source)):
            if mapping.key == key_to_unset:
                removed = True
            else:
                dest.write(mapping.original.string)

    if not removed:
        logger.warning(
            "Key %s not removed from %s - key doesn't exist.", key_to_unset, dotenv_path
        )
        return None, key_to_unset

    return removed, key_to_unset


def _referenced_names(value: Optional[str]) -> Iterator[str]:
    """Yield the names of the variables a raw value references."""
    if value is None:
        return
    for atom in parse_variables(value):
        if isinstance(atom, Variable):
            yield atom.name


def _strongly_connected_components(
    graph: Mapping[str, Sequence[str]],
) -> List[List[str]]:
    """
    Return the strongly connected components of `graph` (Tarjan, iterative).

    Components are returned in a deterministic order: each one is ordered by
    the position of its nodes in `graph`, and components are ordered by the
    position of their first node. Iterative rather than recursive so that a
    long reference chain in a generated .env can't hit the recursion limit.
    """
    order = {name: index for index, name in enumerate(graph)}
    index_of: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    components: List[List[str]] = []
    counter = 0

    for root in graph:
        if root in index_of:
            continue
        # Each frame is a node plus an iterator over its not-yet-visited
        # successors.
        work: List[Tuple[str, Iterator[str]]] = [(root, iter(graph[root]))]
        index_of[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)

        while work:
            node, successors = work[-1]
            advanced = False
            for successor in successors:
                if successor not in graph:
                    continue
                if successor not in index_of:
                    index_of[successor] = lowlink[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, iter(graph[successor])))
                    advanced = True
                    break
                if successor in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[successor])
            if advanced:
                continue

            work.pop()
            if lowlink[node] == index_of[node]:
                component: List[str] = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component, key=order.__getitem__))
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    components.sort(key=lambda component: order[component[0]])
    return components


def _cycle_path(
    component: Sequence[str], graph: Mapping[str, Sequence[str]]
) -> List[str]:
    """
    Return one concrete reference cycle within a strongly connected component.

    The cycle starts and ends with the component's first key (the one that
    appears first in the file) and is found with a breadth-first search, so it
    is also one of the shortest cycles through that key. This keeps the error
    message small even when a large group of keys is tangled together.
    """
    members = set(component)
    start = component[0]
    parents: Dict[str, Optional[str]] = {start: None}
    queue: "deque[str]" = deque([start])

    while queue:
        node = queue.popleft()
        for successor in graph[node]:
            if successor not in members:
                continue
            if successor == start and node != start:
                chain: List[str] = []
                current: Optional[str] = node
                while current is not None:
                    chain.append(current)
                    current = parents[current]
                chain.reverse()
                return [*chain, start]
            if successor not in parents:
                parents[successor] = node
                queue.append(successor)

    # Every node of a strongly connected component can reach the start, so
    # this is unreachable.
    raise AssertionError("no cycle found in a strongly connected component")


def _check_circular_references(values: Iterable[Tuple[str, Optional[str]]]) -> None:
    """
    Raise `CircularInterpolationError` if the file's keys reference each other
    in a cycle.

    The graph has an edge from a key to every other key defined in the file
    that its *effective* definition references. The effective definition is
    the last one for that key: that is the definition whose value is
    delivered, and an earlier definition that was later replaced no longer
    contributes to it.

    Self edges are ignored. `A=${A}` is the supported idiom of falling back to
    the environment, an earlier definition, the default, or an empty string,
    because the key itself is not yet in the values collected from the file
    when it is resolved. A cycle therefore means two or more keys whose final
    values each depend on the other's; neither has a fallback, so whichever is
    resolved first cannot see the other and silently becomes an empty string,
    which is indistinguishable from a legitimately empty value.

    A key that references a key defined later in the file, without any
    reference back, is not a cycle: it keeps the documented behaviour of
    falling back to the environment, the default, or the empty string.
    """
    effective: "OrderedDict[str, Optional[str]]" = OrderedDict()
    for name, value in values:
        effective[name] = value

    graph: "OrderedDict[str, List[str]]" = OrderedDict((name, []) for name in effective)
    for name, value in effective.items():
        for referenced in _referenced_names(value):
            # Only references to other keys defined in this file take part in
            # the cycle; anything else, including the key itself, comes from
            # the environment, an earlier definition, or the default.
            if referenced in graph and referenced != name:
                graph[name].append(referenced)
    for name in graph:
        # A value can reference the same key more than once; keep the graph
        # proportional to the number of distinct references.
        graph[name] = list(dict.fromkeys(graph[name]))

    cycles = [
        _cycle_path(component, graph)
        for component in _strongly_connected_components(graph)
        if len(component) > 1
    ]
    if cycles:
        raise CircularInterpolationError(cycles)


def resolve_variables(
    values: Iterable[Tuple[str, Optional[str]]],
    override: bool,
) -> Mapping[str, Optional[str]]:
    values = list(values)
    _check_circular_references(values)

    new_values: Dict[str, Optional[str]] = {}

    for name, value in values:
        if value is None:
            result = None
        else:
            atoms = parse_variables(value)
            env: Dict[str, Optional[str]] = {}
            if override:
                env.update(os.environ)  # type: ignore
                env.update(new_values)
            else:
                env.update(new_values)
                env.update(os.environ)  # type: ignore
            result = "".join(atom.resolve(env) for atom in atoms)

        new_values[name] = result

    return new_values


def _walk_to_root(path: str) -> Iterator[str]:
    """
    Yield directories starting from the given directory up to the root
    """
    if not os.path.exists(path):
        raise IOError("Starting path not found")

    if os.path.isfile(path):
        path = os.path.dirname(path)

    last_dir = None
    current_dir = os.path.abspath(path)
    while last_dir != current_dir:
        yield current_dir
        parent_dir = os.path.abspath(os.path.join(current_dir, os.path.pardir))
        last_dir, current_dir = current_dir, parent_dir


def find_dotenv(
    filename: str = ".env",
    raise_error_if_not_found: bool = False,
    usecwd: bool = False,
) -> str:
    """
    Search in increasingly higher folders for the given file

    A leading `~` in `filename` is expanded to the user's home directory, and
    an absolute `filename` is returned as is if it exists.

    Returns path to the file if found, or an empty string otherwise
    """
    filename = os.path.expanduser(filename)

    def _is_interactive():
        """Decide whether this is running in a REPL or IPython notebook"""
        if hasattr(sys, "ps1") or hasattr(sys, "ps2"):
            return True
        try:
            main = __import__("__main__", None, None, fromlist=["__file__"])
        except ModuleNotFoundError:
            return False
        return not hasattr(main, "__file__")

    def _is_debugger():
        return sys.gettrace() is not None

    if usecwd or _is_interactive() or _is_debugger() or getattr(sys, "frozen", False):
        # Should work without __file__, e.g. in REPL or IPython notebook.
        path = os.getcwd()
    else:
        # will work for .py files
        frame = sys._getframe()
        current_file = __file__

        while frame.f_code.co_filename == current_file or not os.path.exists(
            frame.f_code.co_filename
        ):
            assert frame.f_back is not None
            frame = frame.f_back
        frame_filename = frame.f_code.co_filename
        path = os.path.dirname(os.path.abspath(frame_filename))

    for dirname in _walk_to_root(path):
        check_path = os.path.join(dirname, filename)
        if _is_file_or_fifo(check_path):
            return check_path

    if raise_error_if_not_found:
        raise IOError("File not found")

    return ""


def load_dotenv(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    override: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
) -> bool:
    """Parse a .env file and then load all the variables found as environment variables.

    Parameters:
        dotenv_path: Absolute or relative path to .env file. A leading `~` is
            expanded to the user's home directory.
        stream: Text stream (such as `io.StringIO`) with .env content, used if
            `dotenv_path` is `None`.
        verbose: Whether to output a warning the .env file is missing.
        override: Whether to override the system environment variables with the variables
            from the `.env` file.
        interpolate: Whether to interpolate variables using POSIX variable expansion.
        encoding: Encoding to be used to read the file.
    Returns:
        Bool: True if at least one environment variable is set else False

    If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
    .env file with its default parameters. If you need to change the default parameters
    of `find_dotenv()`, you can explicitly call `find_dotenv()` and pass the result
    to this function as `dotenv_path`.

    If the environment variable `PYTHON_DOTENV_DISABLED` is set to a truthy value,
    .env loading is disabled.
    """
    if _load_dotenv_disabled():
        logger.debug(
            "python-dotenv: .env loading disabled by PYTHON_DOTENV_DISABLED environment variable"
        )
        return False

    if dotenv_path is None and stream is None:
        dotenv_path = find_dotenv()

    dotenv = DotEnv(
        dotenv_path=dotenv_path,
        stream=stream,
        verbose=verbose,
        interpolate=interpolate,
        override=override,
        encoding=encoding,
    )
    return dotenv.set_as_environment_variables()


def dotenv_values(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
) -> Dict[str, Optional[str]]:
    """
    Parse a .env file and return its content as a dict.

    The returned dict will have `None` values for keys without values in the .env file.
    For example, `foo=bar` results in `{"foo": "bar"}` whereas `foo` alone results in
    `{"foo": None}`

    Parameters:
        dotenv_path: Absolute or relative path to the .env file. A leading `~` is
            expanded to the user's home directory.
        stream: `StringIO` object with .env content, used if `dotenv_path` is `None`.
        verbose: Whether to output a warning if the .env file is missing.
        interpolate: Whether to interpolate variables using POSIX variable expansion.
        encoding: Encoding to be used to read the file.

    If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
    .env file.
    """
    if dotenv_path is None and stream is None:
        dotenv_path = find_dotenv()

    return DotEnv(
        dotenv_path=dotenv_path,
        stream=stream,
        verbose=verbose,
        interpolate=interpolate,
        override=True,
        encoding=encoding,
    ).dict()


def _is_file_or_fifo(path: StrPath) -> bool:
    """
    Return True if `path` exists and is either a regular file or a FIFO.
    """
    if os.path.isfile(path):
        return True

    try:
        st = os.stat(path)
    except (FileNotFoundError, OSError):
        return False

    return stat.S_ISFIFO(st.st_mode)
