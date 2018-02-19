-- Table: public.losses

-- DROP TABLE public.losses;

CREATE TABLE public.losses
(
    id bigint NOT NULL DEFAULT nextval('losses_id_seq'::regclass),
    experiment_id bigint NOT NULL,
    epoch bigint NOT NULL,
    iteration bigint NOT NULL,
    loss real NOT NULL,
    training boolean NOT NULL,
    image text COLLATE pg_catalog."default" NOT NULL,
    flip boolean NOT NULL,
    rotation real NOT NULL,
    size_idx bigint NOT NULL,
    cut_x bigint NOT NULL,
    cut_y bigint NOT NULL,
    CONSTRAINT losses_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.losses
    OWNER to postgres;