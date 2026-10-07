-- Olist e-commerce schema
-- Parent tables first, then child tables (foreign keys need parents to exist)

CREATE TABLE category_translation (
    product_category_name          TEXT PRIMARY KEY,
    product_category_name_english  TEXT
);

CREATE TABLE customers (
    customer_id               TEXT PRIMARY KEY,
    customer_unique_id        TEXT NOT NULL,
    customer_zip_code_prefix  TEXT,
    customer_city             TEXT,
    customer_state            CHAR(2)
);

CREATE TABLE sellers (
    seller_id               TEXT PRIMARY KEY,
    seller_zip_code_prefix  TEXT,
    seller_city             TEXT,
    seller_state            CHAR(2)
);

-- No foreign key to category_translation, because some categories have no translation
CREATE TABLE products (
    product_id                  TEXT PRIMARY KEY,
    product_category_name       TEXT,
    product_name_length         INT,
    product_description_length  INT,
    product_photos_qty          INT,
    product_weight_g            INT,
    product_length_cm           INT,
    product_height_cm           INT,
    product_width_cm            INT
);

CREATE TABLE orders (
    order_id                       TEXT PRIMARY KEY,
    customer_id                    TEXT REFERENCES customers(customer_id),
    order_status                   TEXT,
    order_purchase_timestamp       TIMESTAMP,
    order_approved_at              TIMESTAMP,
    order_delivered_carrier_date   TIMESTAMP,
    order_delivered_customer_date  TIMESTAMP,
    order_estimated_delivery_date  TIMESTAMP
);

CREATE TABLE order_items (
    order_id             TEXT REFERENCES orders(order_id),
    order_item_id        INT,
    product_id           TEXT REFERENCES products(product_id),
    seller_id            TEXT REFERENCES sellers(seller_id),
    shipping_limit_date  TIMESTAMP,
    price                NUMERIC(12,2),
    freight_value        NUMERIC(12,2),
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
    order_id              TEXT REFERENCES orders(order_id),
    payment_sequential    INT,
    payment_type          TEXT,
    payment_installments  INT,
    payment_value         NUMERIC(12,2),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE order_reviews (
    review_id                TEXT,
    order_id                 TEXT REFERENCES orders(order_id),
    review_score             INT,
    review_comment_title     TEXT,
    review_comment_message   TEXT,
    review_creation_date     TIMESTAMP,
    review_answer_timestamp  TIMESTAMP,
    PRIMARY KEY (review_id, order_id)
);

-- Many rows per zip prefix, so this gets a surrogate key and no foreign keys
CREATE TABLE geolocation (
    id                           SERIAL PRIMARY KEY,
    geolocation_zip_code_prefix  TEXT,
    geolocation_lat              NUMERIC(10,6),
    geolocation_lng              NUMERIC(10,6),
    geolocation_city             TEXT,
    geolocation_state            CHAR(2)
);

-- Indexes on columns we join and filter on often
CREATE INDEX idx_orders_customer   ON orders(customer_id);
CREATE INDEX idx_orders_purchase   ON orders(order_purchase_timestamp);
CREATE INDEX idx_items_product     ON order_items(product_id);
CREATE INDEX idx_items_seller      ON order_items(seller_id);
CREATE INDEX idx_geo_zip           ON geolocation(geolocation_zip_code_prefix);