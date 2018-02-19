-- Table: public.accuracies

-- DROP TABLE public.accuracies;

CREATE TABLE public.accuracies
(
    id bigint NOT NULL DEFAULT nextval('accuracies_id_seq'::regclass),
    experiment_id bigint NOT NULL,
    epoch bigint NOT NULL,
    iteration bigint NOT NULL,
    accuracy real NOT NULL,
    training boolean NOT NULL,
    image text COLLATE pg_catalog."default" NOT NULL,
    flip boolean NOT NULL,
    rotation real NOT NULL,
    size_idx bigint NOT NULL,
    cut_x bigint NOT NULL,
    cut_y bigint NOT NULL,
    CONSTRAINT accuracies_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.accuracies
    OWNER to postgres;