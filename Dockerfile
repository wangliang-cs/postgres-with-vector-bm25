# The default Postgres major version
ARG PG_VERSION_MAJOR=17
ARG PGVECTOR_VERSION=v0.8.0
ARG PGSEARCH_VERSION=v0.15.18

# First Stage: Builder
FROM postgres:${PG_VERSION_MAJOR}-bookworm AS builder

ARG PG_VERSION_MAJOR
ARG RUST_VERSION=stable
ARG DEBIAN_FRONTEND=noninteractive

# Declare buildtime environment variables
ENV PG_VERSION_MAJOR=${PG_VERSION_MAJOR} \
    RUST_VERSION=${RUST_VERSION} \
    PATH="/usr/local/bin:/root/.cargo/bin:$PATH" \
    PGX_HOME=/usr/lib/postgresql/${PG_VERSION_MAJOR}

RUN echo "Hello, 0"

# Install build dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    git \
    make \
    gcc \
    clang \
    pkg-config \
    libopenblas-dev \
    postgresql-server-dev-${PG_VERSION_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

# Install Rust with retry for network resilience
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- --default-toolchain "${RUST_VERSION}" -y

# pgvector build stage
FROM builder AS builder-pgvector

ARG PGVECTOR_VERSION
WORKDIR /tmp
RUN git clone --branch ${PGVECTOR_VERSION} --depth 1 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    echo "trusted = true" >> vector.control && \
    make clean && \
    make USE_PGXS=1 OPTFLAGS="" -j

# pg_search build stage
FROM builder AS builder-pg_search

ARG PGSEARCH_VERSION
ARG PG_VERSION_MAJOR

# Install ICU build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the project
RUN git clone --branch ${PGSEARCH_VERSION} --depth 1 https://github.com/paradedb/paradedb.git /tmp/paradedb

# Install pgrx
WORKDIR /tmp/paradedb
RUN PGRX_VERSION=$(cargo tree --depth 1 -i pgrx -p pg_search | head -n 1 | sed -E 's/.*v([0-9]+\.[0-9]+\.[0-9]+).*/\1/') && \
    cargo install --locked cargo-pgrx --version "${PGRX_VERSION}" && \
    cargo pgrx init "--pg${PG_VERSION_MAJOR}=/usr/lib/postgresql/${PG_VERSION_MAJOR}/bin/pg_config"

# Build the extension
WORKDIR /tmp/paradedb/pg_search
RUN cargo pgrx package --features icu --pg-config "/usr/lib/postgresql/${PG_VERSION_MAJOR}/bin/pg_config"

# Final stage
FROM postgres:${PG_VERSION_MAJOR}-bookworm

ARG PG_VERSION_MAJOR

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas3 \
    libicu72 \
    && rm -rf /var/lib/apt/lists/*

# Copy extensions from build stages
COPY --from=builder-pgvector /tmp/pgvector/*.so /usr/lib/postgresql/${PG_VERSION_MAJOR}/lib/
COPY --from=builder-pgvector /tmp/pgvector/*.control /usr/share/postgresql/${PG_VERSION_MAJOR}/extension/
COPY --from=builder-pgvector /tmp/pgvector/sql/*.sql /usr/share/postgresql/${PG_VERSION_MAJOR}/extension/

COPY --from=builder-pg_search /tmp/paradedb/target/release/pg_search-pg${PG_VERSION_MAJOR}/usr/lib/postgresql/${PG_VERSION_MAJOR}/lib/* /usr/lib/postgresql/${PG_VERSION_MAJOR}/lib/plugins/
COPY --from=builder-pg_search /tmp/paradedb/target/release/pg_search-pg${PG_VERSION_MAJOR}/usr/share/postgresql/${PG_VERSION_MAJOR}/extension/* /usr/share/postgresql/${PG_VERSION_MAJOR}/extension/

# Initialize database with extensions
RUN sed -i "s#^module_pathname = .*#module_pathname = '\$libdir/plugins/pg_search'#" /usr/share/postgresql/${PG_VERSION_MAJOR}/extension/pg_search.control && \
    sed -i "s|^#shared_preload_libraries = ''|shared_preload_libraries = '\$libdir/plugins/pg_search'|" /usr/share/postgresql/postgresql.conf.sample

# Set up entrypoint to create extensions
COPY docker-entrypoint-initdb.d /docker-entrypoint-initdb.d
