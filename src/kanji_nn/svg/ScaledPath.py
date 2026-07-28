import numpy as np

class ScaledPath:
    def __init__(self, parsed_path, error=1e-5, scale_factor=109.0):
        """
        Wraps a parsed svg.path object to automatically scale lengths
        and coordinates down by a given factor.

        :param parsed_path: The Path object returned by svg.path.parse_path
        :param error: The approximation error tolerance used for path.length()
        :param scale_factor: The denominator used to scale down coordinates/lengths
        """
        self.path = parsed_path
        self.error = error
        self.scale_factor = scale_factor

        # Cache the underlying raw unscaled length for calculation speed
        self._raw_length = self.path.length(error=self.error)

    def length(self):
        """
        Returns the total length of the SVG path scaled down by the scale factor.
        """
        return self._raw_length / self.scale_factor

    def point(self, t):
        """
        Calculates the point on the path at parameter t (0.0 to 1.0).
        Converts the native complex number point to a scaled NumPy ndarray [x, y].

        :param t: Float between 0.0 and 1.0
        :return: np.ndarray of shape (2,) representing [x, y]
        """
        # Ensure t stays within bounds
        t = max(0.0, min(1.0, float(t)))

        # Get raw complex point from native library
        raw_point = self.path.point(t)

        # Extract components, convert to ndarray, and scale down
        scaled_x = raw_point.real / self.scale_factor
        scaled_y = raw_point.imag / self.scale_factor

        return np.array([scaled_x, scaled_y], dtype=np.float64)

    def interpolate_uniform(self, n_out):
        """
        Resamples the path into n_out points distributed evenly by ARCLENGTH.
        Automatically tracks which original segment index produced each point.

        :param n_out: Integer count of output vertices
        :return: A tuple containing:
                 - vertices: np.ndarray of shape (n_out, 2)
                 - segment_indices: np.ndarray of shape (n_out,) matching vertex indices to segment indices
        """
        if n_out < 2:
            raise ValueError("n_out must be at least 2 to interpolate a path.")

        # 1. Determine equidistant distances along the raw, unscaled path length
        target_lengths = np.linspace(0, self._raw_length, n_out)

        # 2. Gather individual raw segment lengths
        segment_lengths = [seg.length(error=self.error) for seg in self.path]
        cum_lengths = np.cumsum([0.0] + segment_lengths)

        # Pre-allocate both the coordinates array and the segment index map array
        vertices = np.zeros((n_out, 2), dtype=np.float64)
        segment_indices = np.zeros(n_out, dtype=np.int32)

        seg_idx = 0

        # 3. Step through target lengths and calculate exact points + track segments
        for i, length in enumerate(target_lengths):
            # Advance to the correct segment matching the current distance
            while seg_idx < len(self.path) - 1 and length > cum_lengths[seg_idx + 1]:
                seg_idx += 1

            seg_start_len = cum_lengths[seg_idx]
            seg_total_len = segment_lengths[seg_idx]

            # Map absolute length to the localized parametric segment t (0.0 to 1.0)
            if seg_total_len > 0:
                local_t = (length - seg_start_len) / seg_total_len
            else:
                local_t = 0.0

            # Get native point from the specific sub-segment component
            raw_point = self.path[seg_idx].point(local_t)

            # Populate coordinate array
            vertices[i, 0] = raw_point.real / self.scale_factor
            vertices[i, 1] = raw_point.imag / self.scale_factor

            # Populate reverse lookup mapping array
            segment_indices[i] = seg_idx

        return vertices, segment_indices

    def interpolate_naive(self, n_out):
        """
        Resamples the path using raw parametric steps.
        NOTE: Vertices will NOT be perfectly equidistant by distance.
        """
        if n_out < 2:
            raise ValueError("n_out must be at least 2.")

        # Linearly space t from 0.0 to 1.0
        t_values = np.linspace(0.0, 1.0, n_out)

        # Pull points directly from the main path object
        vertices = np.zeros((n_out, 2), dtype=np.float64)
        for i, t in enumerate(t_values):
            raw_point = self.path.point(t)
            vertices[i, 0] = raw_point.real / self.scale_factor
            vertices[i, 1] = raw_point.imag / self.scale_factor

        return vertices

    def __len__(self):
        """Allows querying how many sub-segments the path contains."""
        return len(self.path)

    def __getitem__(self, index):
        """Allows direct indexing of underlying segments if needed."""
        return self.path[index]
